from PIL import ImageFile
import collections
from copy import deepcopy
from datasets import load_from_disk, set_caching_enabled
from scipy.optimize import linear_sum_assignment
from utils import data_utils, utils
from utils.args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
# from transformers.models.detr.modeling_detr import DetrHungarianMatcher
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from simmc2.model.utils import ambiguous_candidates_evaluation as eval_utils

import datasets
import json
import logging
import numpy as np
import os
import pandas as pd
import sys
import torch
import torch.nn as nn
import torchvision.transforms as T
import transformers

set_caching_enabled(True)
logger = logging.getLogger(__name__)

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
        (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def center_to_corners_format(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union

def requires_backends(obj, backends):
    if not isinstance(backends, (list, tuple)):
        backends = [backends]
    name = obj.__name__ if hasattr(obj, "__name__") else obj.__class__.__name__

def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/. The boxes should be in [x0, y0, x1, y1] (corner) format.
    Returns:
        `torch.FloatTensor`: a [N, M] pairwise matrix, where N = len(boxes1) and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    if not (boxes1[:, 2:] >= boxes1[:, :2]).all():
        raise ValueError(f"boxes1 must be in [x0, y0, x1, y1] (corner) format, but got {boxes1}")
    if not (boxes2[:, 2:] >= boxes2[:, :2]).all():
        raise ValueError(f"boxes2 must be in [x0, y0, x1, y1] (corner) format, but got {boxes2}")
    # print(boxes1.shape, boxes2.shape)
    iou, union = box_iou(boxes1, boxes2)

    top_left = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    width_height = (bottom_right - top_left).clamp(min=0)  # [N,M,2]
    area = width_height[:, :, 0] * width_height[:, :, 1]

    return iou - (area - union) / area


#####
# Main Functions
#####
split_name = 'test'

def run(model_args, data_args, training_args):
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)
    os.makedirs(training_args.output_dir, exist_ok=True)
    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Data loading
    MAPPING = data_utils.load_categories()

    conv_train_dset, train_gold_data = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.train_dataset_path,
        mapping=MAPPING, return_gt_labels=True
    )
    conv_dev_dset, valid_gold_data = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.dev_dataset_path,
        mapping=MAPPING, return_gt_labels=True
    )
    conv_test_dset, test_gold_data = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.devtest_dataset_path,
        mapping=MAPPING, return_gt_labels=True
    )
    # conv_test_dset = conv_test_dset.shard(num_shards=10, index=5)
    
    # Preprocessing
    if data_args.augment_with_scene_data:
        scene_dset, MAPPING = data_utils.load_objects_in_scenes_dataset(mapping=MAPPING)    
        scene_dset = scene_dset.map(
            data_utils.add_sitcom_detr_attr,
            num_proc=data_args.preprocessing_num_workers,
            desc="adding sitcom detr attribute",
            load_from_cache_file=False,
            remove_columns=None
        )

        dataset = datasets.DatasetDict({
            'train': datasets.concatenate_datasets([scene_dset, conv_train_dset]), 
            'valid': conv_dev_dset, 
            'test': conv_test_dset,
        })
    else:
        dataset = datasets.DatasetDict({
            'train': conv_train_dset, 
            'valid': conv_dev_dset, 
            'test': conv_test_dset,
        })
        
    dataset = dataset.map(
        data_utils.convert_dialogue_to_caption,
        num_proc=data_args.preprocessing_num_workers,
        desc="convert object attributes to caption",
        load_from_cache_file=False,
        fn_kwargs={"num_utterances": data_args.num_utterances},
        remove_columns=["dialogue"]
    )

    # def transform(example_batch):
    #     images = [image.convert("RGB") for image in example_batch["image"]]
        
    #     # Preprocess target objects
    #     targets = [
    #         {"image_id": id_, "annotations": object_} \
    #         for (id_, object_) in zip(example_batch["image_id"], example_batch["objects"])
    #     ]
    #     features = feature_extractor(images=images, annotations=targets, return_tensors="pt")
    #     for key, value in features.items():
    #         example_batch[key] = value

    #     for i, object_ in enumerate(example_batch["objects"]):
    #         example_batch['labels'][i]['turn_id'] = torch.LongTensor([example_batch['turn_id'][i]])
    #         example_batch['labels'][i]['dialog_id'] = torch.LongTensor([example_batch['dialog_id'][i]])
    #         example_batch['labels'][i]['index'] = torch.LongTensor(list(map(lambda x: x['index'], object_)))
            
    #     # Preprocess all objects
    #     all_targets = [
    #         {"image_id": idx, "annotations": object_} \
    #         for idx, object_ in enumerate(example_batch["all_objects"])
    #     ]
    #     features = feature_extractor(images=images, annotations=all_targets, return_tensors="pt")
    #     for key in features['labels'][0].keys():
    #         for i in range(len(features['labels'])):
    #             example_batch['labels'][i][f"all_{key}"] = features['labels'][i][key]

    #     for i, object_ in enumerate(example_batch["all_objects"]):
    #         example_batch['labels'][i]['all_index'] = torch.LongTensor(list(map(lambda x: x['index'], object_)))
            
    #     return example_batch
    
    # proc_datasets = deepcopy(dataset)
    # proc_datasets["train"] = proc_datasets["train"].with_transform(transform)
    # proc_datasets["valid"] = proc_datasets["valid"].with_transform(transform)
    # proc_datasets["test"] = proc_datasets["test"].with_transform(transform)
    
    model, postprocessor = torch.hub.load('ashkamath/mdetr:main', 'mdetr_efficientnetB5', pretrained=True, return_postprocessor=True)
    model = model.cuda()

    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # class DetrHungarianMatcher(nn.Module):
    #     """
    #     This class computes an assignment between the targets and the predictions of the network.
    #     For efficiency reasons, the targets don't include the no_object. Because of this, in general, there are more
    #     predictions than targets. In this case, we do a 1-to-1 matching of the best predictions, while the others are
    #     un-matched (and thus treated as non-objects).
    #     Args:
    #         class_cost:
    #             The relative weight of the classification error in the matching cost.
    #         bbox_cost:
    #             The relative weight of the L1 error of the bounding box coordinates in the matching cost.
    #         giou_cost:
    #             The relative weight of the giou loss of the bounding box in the matching cost.
    #     """

    #     def __init__(self, class_cost: float = 1, bbox_cost: float = 1, giou_cost: float = 1):
    #         super().__init__()
    #         requires_backends(self, ["scipy"])

    #         self.class_cost = class_cost
    #         self.bbox_cost = bbox_cost
    #         self.giou_cost = giou_cost
    #         if class_cost == 0 and bbox_cost == 0 and giou_cost == 0:
    #             raise ValueError("All costs of the Matcher can't be 0")

    #     @torch.no_grad()
    #     def forward(self, outputs, targets):
    #         """
    #         Args:
    #             outputs (`dict`):
    #                 A dictionary that contains at least these entries:
    #                 * "logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
    #                 * "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates.
    #             targets (`List[dict]`):
    #                 A list of targets (len(targets) = batch_size), where each target is a dict containing:
    #                 * "class_labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of
    #                 ground-truth
    #                 objects in the target) containing the class labels
    #                 * "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates.
    #         Returns:
    #             `List[Tuple]`: A list of size `batch_size`, containing tuples of (index_i, index_j) where:
    #             - index_i is the indices of the selected predictions (in order)
    #             - index_j is the indices of the corresponding selected targets (in order)
    #             For each batch element, it holds: len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
    #         """
    #         batch_size, num_queries = outputs["logits"].shape[:2]

    #         # We flatten to compute the cost matrices in a batch
    #         out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
    #         out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

    #         # Also concat the target labels and boxes
    #         target_ids = torch.cat([v["class_labels"] for v in targets])
    #         target_bbox = torch.cat([v["boxes"] for v in targets])

    #         # Compute the classification cost. Contrary to the loss, we don't use the NLL,
    #         # but approximate it in 1 - proba[target class].
    #         # The 1 is a constant that doesn't change the matching, it can be ommitted.
    #         class_cost = -out_prob[:, target_ids]

    #         # Compute the L1 cost between boxes
    #         bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)

    #         # Compute the giou cost between boxes
    #         giou_cost = -generalized_box_iou(center_to_corners_format(out_bbox), center_to_corners_format(target_bbox))

    #         # Final cost matrix
    #         cost_matrix = self.bbox_cost * bbox_cost + self.class_cost * class_cost + self.giou_cost * giou_cost
    #         cost_matrix = cost_matrix.view(batch_size, num_queries, -1).cpu()

    #         sizes = [len(v["boxes"]) for v in targets]
    #         indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
    #         return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    # matcher = DetrHungarianMatcher(
    #     class_cost=1,
    #     bbox_cost=1, 
    #     giou_cost=1,
    # )

    class HungarianMatcher(nn.Module):
        """This class computes an assignment between the targets and the predictions of the network
        For efficiency reasons, the targets don't include the no_object. Because of this, in general,
        there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
        while the others are un-matched (and thus treated as non-objects).
        """

        def __init__(self, cost_class: float = 0, cost_bbox: float = 1, cost_giou: float = 1):
            """Creates the matcher
            Params:
                cost_class: This is the relative weight of the classification error in the matching cost
                cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
                cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
            """
            super().__init__()
            self.cost_class = cost_class
            self.cost_bbox = cost_bbox
            self.cost_giou = cost_giou
            self.norm = nn.Softmax(-1)
            assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

        @torch.no_grad()
        def forward(self, outputs, targets, positive_map=None):
            """Performs the matching
            Params:
                outputs: This is a dict that contains at least these entries:
                    "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                    "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
                targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                    "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                            objects in the target) containing the class labels
                    "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
            Returns:
                A list of size batch_size, containing tuples of (index_i, index_j) where:
                    - index_i is the indices of the selected predictions (in order)
                    - index_j is the indices of the corresponding selected targets (in order)
                For each batch element, it holds:
                    len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
            """
            bs = 1
            num_queries = outputs["pred_logits"].shape[0]

            # We flatten to compute the cost matrices in a batch
            out_prob = self.norm(outputs["pred_logits"]) #.flatten(0, 1))  # [batch_size * num_queries, num_classes]
            # print("before", outputs["pred_boxes"].shape)
            out_bbox = outputs["pred_boxes"] #.flatten(0, 1)  # [batch_size * num_queries, 4]
            # print("after", out_bbox.shape)

            # Also concat the target labels and boxes
            tgt_ids = torch.stack([torch.tensor(v["category_id"]) for v in targets], dim=0)
            tgt_bbox = torch.stack([torch.tensor(v["bbox"]) for v in targets], dim=0)
            # assert len(tgt_bbox) == len(positive_map)

            # print("out", out_bbox, "tgt", tgt_bbox)
            # print()

            # Compute the soft-cross entropy between the predicted token alignment and the GT one for each box
            # cost_class = -(out_prob.unsqueeze(1) * positive_map.unsqueeze(0)).sum(-1)
            # cost_class = 0
            cost_class = -out_prob[:, tgt_ids]

            # Compute the L1 cost between boxes
            cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
            # assert cost_class.shape == cost_bbox.shape

            # Compute the giou cost betwen boxes
            cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox))

            # Final cost matrix
            C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
            C = C.view(bs, num_queries, -1).cpu()

            sizes = [len(targets)]
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)

    def inference(im, caption, all_objects, model=model, batch_size=128, transform=transform, matcher=matcher):
        # mean-std normalize the input image (batch-size: 1)
        img = transform(im).unsqueeze(0).cuda()

        # propagate through the model
        memory_cache = model(img, [caption], encode_and_save=True)
        outputs = model(img, [caption], encode_and_save=False, memory_cache=memory_cache)
        #   print(outputs)

        # keep only predictions with 0.7+ confidence
        probas = 1 - outputs['pred_logits'].softmax(-1)[0, :, -1].cpu()
        keep = (probas > 0.5).cpu()
        # logits = outputs['pred_logits'].cpu()[0, keep]
        logits = outputs['pred_logits'].cpu()[0, :]

        # # convert boxes from [0; 1] to image scales
        def rescale_bboxes(out_bbox, size):
            img_w, img_h = size
            b = box_cxcywh_to_xyxy(out_bbox)
            b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
            return b
        # bboxes = rescale_bboxes(outputs['pred_boxes'].cpu()[0, keep], im.size)
        bboxes = rescale_bboxes(outputs['pred_boxes'].cpu()[0, :], im.size)
        # bboxes = outputs['pred_boxes'].cpu()[0, :]

        # Extract the text spans predicted by each box
        positive_tokens = (logits.softmax(-1) > 0.1).nonzero().tolist()
        predicted_spans = collections.defaultdict(str)
        for tok in positive_tokens:
            item, pos = tok
            if pos < 255:
                try:
                    span = memory_cache["tokenized"].token_to_chars(0, pos)
                    predicted_spans[item] += " " + caption[span.start:span.end]
                except:
                    predicted_spans[item] += ""
                # print("span", span, "item", item)
                # print("predicted_spans[item]", predicted_spans[item])
        labels = [predicted_spans[k] if k in predicted_spans else "" for k in range(len(logits))]

        # print("output", bboxes)
        
        batch_outputs = {
            'pred_logits': logits,
            'pred_boxes': bboxes
        }
        batch_targets = all_objects
        # print("target", batch_targets)
        # quit()
        # print(batch_outputs, batch_targets)
        match_indices = matcher(batch_outputs, batch_targets)
        
        pred_indices, gt_indices = match_indices[0]
        
        iou_scores, _ = box_iou(
            center_to_corners_format(torch.tensor(bboxes)[pred_indices]),
            center_to_corners_format(torch.tensor([obj["bbox"] for obj in all_objects])[gt_indices])
        )
        iou_scores  = iou_scores.diagonal()
        valid_boxes = (iou_scores >= 0.1)
        pred_obj_ids = []
        pred_labels = []
        for j in range(len(valid_boxes)):
            if valid_boxes[j]:
                # print(len(valid_boxes), len(labels), len(pred_indices), j, pred_indices[j])
                pred_obj_ids.append(all_objects[gt_indices[j]]['index'])
                pred_labels.append(labels[pred_indices[j]])

        # pred_obj_ids = []
        # # print("all_objects", len(all_objects), all_objects)
        # # print("keep", keep)
        # # print("gt_indices", gt_indices)
        # # print("pred_indices", pred_indices)
        # for j, pred_id in enumerate(pred_indices):
        #     if keep[pred_id]:
        #         pred_obj_ids.append(all_objects[gt_indices[j]]['index'])

        return pred_obj_ids, pred_labels
    
    results = collections.defaultdict(list)
    for i, batch in tqdm(enumerate(dataset[split_name]), total=len(dataset[split_name])):
        pred_obj_ids, pred_labels = inference(
            im=batch["image"].convert("RGB"), caption=batch["caption"], all_objects=batch["all_objects"])
        results[batch["dialog_id"]].append({
            "turn_id": batch["turn_id"],
            "disambiguation_candidates": pred_obj_ids,
            "disambiguation_labels": pred_labels,
        })

    # Restructure results JSON and save.
    print('Comparing predictions with ground truths...')
    results = [{
        "dialog_id": dialog_id,
        "predictions": predictions,
    } for dialog_id, predictions in results.items()]

    if split_name == 'train':
        gold_data = train_gold_data
    elif split_name == 'valid': 
        gold_data = valid_gold_data
    elif split_name == 'test':
        gold_data = test_gold_data 
    else:
        raise ValueError(f'Unknown split name `{split_name}`')
    metrics = eval_utils.evaluate_ambiguous_candidates(gold_data, results)

    print(f'== Eval Metrics ==')
    print('Recall: ', metrics["recall"])
    print('Precision: ', metrics["precision"])
    print('F1-Score: ', metrics["f1"])


def main():
    ###
    # Parsing & Initialization
    ###
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Set random seed
    utils.init_env(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    # Init logging
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    ###
    # RUN RUN RUN!!!
    ###
    run(model_args, data_args, training_args)
    
if __name__ == '__main__':
    main()