from PIL import ImageFile
import collections
from copy import deepcopy
from datasets import load_from_disk, set_caching_enabled
from detr import CocoEvaluator
from utils import data_utils, utils
from utils.args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from tqdm import tqdm
from trainer.detr_trainer import DetrTrainer
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
    EarlyStoppingCallback
)
from transformers.models.detr.modeling_detr import DetrHungarianMatcher
from transformers import HfArgumentParser, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from typing import Dict, Union, Any, Optional, List, Tuple
from model.holy_detr import HolyDetrForObjectDetection
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
import transformers

set_caching_enabled(True)
logger = logging.getLogger(__name__)


#####
# Main Functions
#####
is_test = False
def run(model_args, data_args, training_args):
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)
    os.makedirs(training_args.output_dir, exist_ok=True)
    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Data loading
    MAPPING = data_utils.load_categories()

    conv_train_dset = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.train_dataset_path,
        mapping=MAPPING, return_gt_labels=False
    )
    conv_dev_dset, valid_gold_data = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.dev_dataset_path,
        mapping=MAPPING, return_gt_labels=True
    )
    conv_test_dset, test_gold_data = data_utils.load_sitcom_detr_dataset(
        data_path=data_args.devtest_dataset_path,
        mapping=MAPPING, return_gt_labels=True
    )
    
    # Preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.text_model_name_or_path)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    if data_args.augment_with_scene_data:
        scene_dset, MAPPING = data_utils.load_objects_in_scenes_dataset(mapping=MAPPING)    
        scene_dset = scene_dset.map(
            data_utils.add_sitcom_detr_attr,
            num_proc=data_args.preprocessing_num_workers,
            desc="adding sitcom detr attribute",
            load_from_cache_file=True,
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
    
    dataset = dataset.map(
        data_utils.tokenize_text,
        num_proc=data_args.preprocessing_num_workers,
        desc="tokenize text data",
        load_from_cache_file=False,
        fn_kwargs={"tokenizer": tokenizer, "text_column_name": "caption"},
        remove_columns=["caption"]
    )
    
    def transform(example_batch):
        images = [image.convert("RGB") for image in example_batch["image"]]
        
        # Preprocess target objects
        targets = [
            {"image_id": id_, "annotations": object_} \
            for (id_, object_) in zip(example_batch["image_id"], example_batch["objects"])
        ]
        features = feature_extractor(images=images, annotations=targets, return_tensors="pt")
        for key, value in features.items():
            example_batch[key] = value

        for i, object_ in enumerate(example_batch["objects"]):
            example_batch['labels'][i]['turn_id'] = torch.LongTensor([example_batch['turn_id'][i]])
            example_batch['labels'][i]['dialog_id'] = torch.LongTensor([example_batch['dialog_id'][i]])
            example_batch['labels'][i]['index'] = torch.LongTensor(list(map(lambda x: x['index'], object_)))
            
        # Preprocess all objects
        all_targets = [
            {"image_id": idx, "annotations": object_} \
            for idx, object_ in enumerate(example_batch["all_objects"])
        ]
        features = feature_extractor(images=images, annotations=all_targets, return_tensors="pt")
        for key in features['labels'][0].keys():
            for i in range(len(features['labels'])):
                example_batch['labels'][i][f"all_{key}"] = features['labels'][i][key]
                
        return example_batch
    
    proc_datasets = deepcopy(dataset)
    proc_datasets["train"] = proc_datasets["train"].with_transform(transform)
    proc_datasets["valid"] = proc_datasets["valid"].with_transform(transform)
    proc_datasets["test"] = proc_datasets["test"].with_transform(transform)
    
    # Training and evaluation
    text_collator = DataCollatorWithPadding(tokenizer)
    
    def collate_fn(batch):
        pixel_values = [item["pixel_values"] for item in batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(
            pixel_values, return_tensors="pt"
        )
        labels = [item["labels"] for item in batch]
        text_batch = text_collator({'input_ids': [item["input_ids"] for item in batch]})
        
        batch = {}
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        batch["input_ids"] = text_batch["input_ids"]
        batch["attention_mask"] = text_batch["attention_mask"]

        return batch

    text_model = transformers.AutoModel.from_pretrained(model_args.text_model_name_or_path)
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        id2label=MAPPING["id2cat"],
        label2id=MAPPING["cat2id"],
    )
    detr_model = transformers.AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        id2label=MAPPING["id2cat"],
        label2id=MAPPING["cat2id"],
        ignore_mismatched_sizes=True
    )
    config.text_auxiliary_loss = False
    holy_detr = HolyDetrForObjectDetection(config, text_model)
    holy_detr.load_state_dict(detr_model.state_dict(), strict=False)
    
    matcher = DetrHungarianMatcher(
        class_cost=holy_detr.config.class_cost, 
        bbox_cost=holy_detr.config.bbox_cost, 
        giou_cost=holy_detr.config.giou_cost
    )
    def compute_metrics(p: EvalPrediction):
        def box_area(boxes):
            return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        def box_iou(boxes1, boxes2):
            area1 = box_area(boxes1)
            area2 = box_area(boxes2)

            left_top = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
            right_bottom = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

            width_height = (right_bottom - left_top).clamp(min=0)  # [N,M,2]
            inter = width_height[:, :, 0] * width_height[:, :, 1]  # [N,M]

            union = area1[:, None] + area2 - inter

            iou = inter / union
            return iou
        
        # p.prediction: Dict{'pred_logits': Tensor, 'pred_boxes': Tensor}
        # p.labels_ids: List[Dict{
        #    'class_labels': Tensor, 'boxes': Tensor, 'image_id': Tensor, 'area': tensor,
        #    'iscrowd': Tensor, 'orig_size': Tensor, 'size': Tensor
        # }]
        labels = p.label_ids
        outputs = p.predictions
        
        all_objects = []
        for label in labels:
            all_object = {}
            for k, v in label.items():
                if 'all_' in k:
                    all_object[k.replace('all_','')] = v
            all_objects.append(all_object)
        
        no_obj_idx = outputs['logits'].shape[-1] # index of no object prediction
        probas = outputs['logits'].softmax(-1)
        cls_preds = probas.argmax(dim=-1)
        boxes_preds = outputs['pred_boxes']
        
        match_indices = matcher(outputs, all_objects) 
        pred_indices = []
                
        # probas: torch.Size([414, 100, 29])
        # cls_preds: torch.Size([414, 100])
        # boxes_preds: torch.Size([414, 100, 4])
        # labels: List[Dict{'class_labels': Tensor, 'boxes': Tensor, 'index': Tensor}] (len 414)
        # all_objects: List[Dict{'class_labels': Tensor, 'boxes'': Tensor, 'index': Tensor}] (len 414)
        # indices: List[Tuple<pred_idxs, gt_idxs>] (len 414)
        results = collections.defaultdict(list)
        for boxes_pred, label, all_object, (pred_idxs, gt_idxs) in zip(boxes_preds, labels, all_objects, match_indices):
            tgt_boxes = all_object['boxes']
            iou_scores = box_iou(boxes_pred[pred_idxs], tgt_boxes[gt_idxs]).diagonal()
            valid_boxes = (iou_scores >= 0.5)
            
            turn_id = label['turn_id'].item()
            dialog_id = label['dialog_id'].item()
            
            pred_obj_ids = []
            for j in range(len(valid_boxes)):
                # nz = multihot_batch.nonzero()
                # nz[nz[:,0] == 0,1]
                if valid_boxes[j]:
                    pred_obj_ids.append(all_objects[gt_idxs[j]]['index'])
            
            new_instance = {
                "turn_id": turn_id,
                "disambiguation_candidates": pred_obj_ids
            }
            results[dialog_id].append(new_instance)

        # Restructure results JSON and save.
        print('Compariong predictions with ground truths...')
        results = [{
            "dialog_id": dialog_id,
            "predictions": predictions,
        } for dialog_id, predictions in results.items()]

        global is_test
        gold_data = test_gold_data if is_test else valid_gold_data
        metrics = eval_utils.evaluate_ambiguous_candidates(gold_data, results)

        print('== Eval Metrics ==')
        print('Recall: ', metrics["recall"])
        print('Precision: ', metrics["precision"])
        print('F1-Score: ', metrics["f1"])

        return metrics

    
    trainer = DetrTrainer(
        model=holy_detr,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=proc_datasets["train"],
        eval_dataset=proc_datasets["valid"],
        compute_metrics=compute_metrics,
        tokenizer=feature_extractor,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Training
    train_results = trainer.train()
    trainer.save_model()

    # Evaluation
    metrics = trainer.evaluate(proc_datasets["valid"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    global is_test
    is_test = True # Nasty Global Variable for Compute Metric
    
    metrics = trainer.evaluate(proc_datasets["test"])
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)
    
    # # Prediction
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # for idx, batch in enumerate(tqdm(proc_datasets["all"])):
    #     # forward pass
    #     outputs = model(
    #         pixel_values=batch["pixel_values"].unsqueeze(dim=0).to(device),
    #         pixel_mask=batch["pixel_mask"].unsqueeze(dim=0).to(device))
    #     print(outputs.pred_boxes.shape, outputs.last_hidden_state)


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