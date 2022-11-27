from PIL import ImageFile
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
from transformers import HfArgumentParser
from transformers.models.detr.modeling_detr import DetrHungarianMatcher
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from typing import Dict, Union, Any, Optional, List, Tuple

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
def run(model_args, data_args, training_args):
    training_args.output_dir="{}/{}".format(training_args.output_dir, model_args.model_name_or_path)
    os.makedirs(training_args.output_dir, exist_ok=True)
    cache_dir_path = "./{}/{}".format(data_args.cache_dir_name, model_args.model_name_or_path)
    os.makedirs(cache_dir_path, exist_ok=True)

    # Data loading
    MAPPING = data_utils.load_categories()
    dataset, MAPPING = data_utils.load_objects_in_scenes_dataset(mapping=MAPPING)
    dataset = dataset.map(
        data_utils.compute_image_area,
        num_proc=data_args.preprocessing_num_workers,
        desc="compute image area",
        load_from_cache_file=True,
        cache_file_name=os.path.join(cache_dir_path, "ds_area.arrow")
    )
    raw_datasets = dataset.train_test_split(0.1)
    raw_datasets["all"] = dataset

    # Preprocessing
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
        model_args.model_name_or_path)

    def transform(example_batch):
        images = [image.convert("RGB") for image in example_batch["image"]]
        targets = [
            {"image_id": id_, "annotations": object_} \
            for (id_, object_) in zip(example_batch["image_id"], example_batch["objects"])
        ]
        batch = feature_extractor(images=images, annotations=targets, return_tensors="pt")
        batch["image_id"] = [image_id for image_id in example_batch["image_id"]]
        batch["image"] = images
        return batch
    proc_datasets = deepcopy(raw_datasets)
    proc_datasets["train"] = proc_datasets["train"].with_transform(transform)
    proc_datasets["test"] = proc_datasets["test"].with_transform(transform)
    proc_datasets["all"] = proc_datasets["all"].with_transform(transform)

    # Training and evaluation
    def collate_fn(example_batch):
        pixel_values = [item["pixel_values"] for item in example_batch]
        encoding = feature_extractor.pad_and_create_pixel_mask(
            pixel_values, return_tensors="pt"
        )
        labels = [item["labels"] for item in example_batch]
        batch = {}
        # batch["image_id"] = [item["image_id"] for item in example_batch]
        batch["pixel_values"] = encoding["pixel_values"]
        batch["pixel_mask"] = encoding["pixel_mask"]
        batch["labels"] = labels
        return batch

    model = transformers.AutoModelForObjectDetection.from_pretrained(
        model_args.model_name_or_path,
        id2label=MAPPING["id2cat"],
        label2id=MAPPING["cat2id"],
        ignore_mismatched_sizes=True)
    
    trainer = DetrTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=proc_datasets["train"],
        eval_dataset=proc_datasets["test"],
        tokenizer=feature_extractor,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # # Evaluation
    # metrics = trainer.evaluate(proc_datasets["test"])
    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)

    # metrics = trainer.evaluate(proc_datasets["all"])
    # trainer.log_metrics("all", metrics)
    # trainer.save_metrics("all", metrics)
    
    # print(proc_datasets["all"][0]
    # )
    # Prediction
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = DetrHungarianMatcher(
        class_cost=model.config.class_cost, bbox_cost=model.config.bbox_cost, giou_cost=model.config.giou_cost)
    pred_results = {}
    for idx, batch in enumerate(tqdm(proc_datasets["all"])):
        # forward pass
        # print(batch["image"].size)
        outputs = model(
            pixel_values=batch["pixel_values"].unsqueeze(dim=0).to(device),
            pixel_mask=batch["pixel_mask"].unsqueeze(dim=0).to(device))
        # target_sizes = torch.tensor([batch["image"].size[::-1]]).to(device)
        # print("target_size", target_sizes)
        # results = feature_extractor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        # print(results["boxes"].shape, batch["labels"]["boxes"].shape)

        # out_logits, out_bbox = outputs.logits, outputs.pred_boxes
        # prob = nn.functional.softmax(out_logits, -1)
        # print("PROB BABI", prob)
        # print(prob.max())
        # print(batch["labels"])
        indices = matcher(
            outputs,
            [{"class_labels": batch["labels"]["class_labels"].to(device), "boxes": batch["labels"]["boxes"].to(device)}])
        # print("MATCHER", indices[0].shape)
        pred_indices, gt_indices = indices[0]
        sorted_pred_indices = torch.tensor([p for _, p in sorted(zip(gt_indices, pred_indices))]).to(device)
        gt_num_objects = batch["labels"]["boxes"].shape[0]
        pred_last_hidden_state = torch.index_select(outputs.last_hidden_state.squeeze(), 0, sorted_pred_indices)
        pred_boxes = torch.index_select(outputs.pred_boxes.squeeze(), 0, sorted_pred_indices)
        pred_results[MAPPING["id2scene"][batch["image_id"]] + ".png"] = torch.cat([
            pred_last_hidden_state.detach().cpu() #, pred_boxes.detach().cpu()
        ], dim=1)
    if not os.path.exists(data_args.preprocessed_data_dir):
        os.makedirs(data_args.preprocessed_data_dir)
    visual_features_path = os.path.join(data_args.preprocessed_data_dir, "visual_features_detr_no_bbox.pt")
    torch.save(pred_results, visual_features_path)
    print(f"Saved to {visual_features_path}.")
    # results = trainer.predict(proc_datasets["all"])


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