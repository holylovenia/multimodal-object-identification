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
from transformers import HfArgumentParser, DataCollatorWithPadding
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from typing import Dict, Union, Any, Optional, List, Tuple
from model.holy_detr import HolyDetrForObjectDetection

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
    scene_dset, MAPPING = data_utils.load_objects_in_scenes_dataset(mapping=MAPPING)    
    
    conv_train_dset = data_utils.load_sitcom_detr_dataset(
        data_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json'
        mapping=MAPPING, return_gt_labels=False
    )
    conv_dev_dset = data_utils.load_sitcom_detr_dataset(
        data_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json'
        mapping=MAPPING, return_gt_labels=False
    )
    conv_test_dset = data_utils.load_sitcom_detr_dataset(
        data_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json'
        mapping=MAPPING, return_gt_labels=False
    )
    
    # Preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.text_model_name_or_path)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)

    scene_dataset = scene_dataset.map(
        data_utils.add_empty_dialogue,
        num_proc=data_args.preprocessing_num_workers,
        desc="adding empty dialogue",
        load_from_cache_file=False,
        cache_file_name=os.path.join(cache_dir_path, "ds_dialogue.arrow"),
        remove_columns=None
    )
    
    dataset = datasets.DatasetDict({
        'train': datasets.concatenate_datasets([scene_dset, conv_train_dset]), 
        'valid': conv_dev_dset, 'test': conv_test_dset
    })
        
    dataset = dataset.map(
        data_utils.compute_image_area,
        num_proc=data_args.preprocessing_num_workers,
        desc="compute image area",
        load_from_cache_file=False,
        cache_file_name=os.path.join(cache_dir_path, "ds_area.arrow"),
        remove_columns=None
    )
    
    dataset = dataset.map(
        data_utils.convert_dialogue_to_caption,
        num_proc=data_args.preprocessing_num_workers,
        desc="convert object attributes to caption",
        load_from_cache_file=False,
        cache_file_name=os.path.join(cache_dir_path, "ds_convert.arrow"),
        fn_kwargs={"num_utterances": data_args.num_utterances},
        remove_columns=["dialogue"]
    )
    
    dataset = dataset.map(
        data_utils.tokenize_text,
        num_proc=data_args.preprocessing_num_workers,
        desc="tokenize text data",
        load_from_cache_file=False,
        cache_file_name=os.path.join(cache_dir_path, "ds_token.arrow"),
        fn_kwargs={"tokenizer": tokenizer, "text_column_name": "caption"},
        remove_columns=["caption"]
    )

    def transform(example_batch):
        images = [image.convert("RGB") for image in example_batch["image"]]
        targets = [
            {"image_id": id_, "annotations": object_} \
            for (id_, object_) in zip(example_batch["image_id"], example_batch["objects"])
        ]
        features = feature_extractor(images=images, annotations=targets, return_tensors="pt")
        for key, value in features.items():
            example_batch[key] = value
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
    
    def compute_metrics():
        for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
            # keep only predictions of queries with 0.9+ confidence (excluding no-object class)
            probas = outputs.logits.softmax(-1)[0, :, :-1]
            keep = probas.max(-1).values >= threshold

            # rescale bounding boxes
            target_sizes = torch.tensor(im.size[::-1]).unsqueeze(0)
            postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
            bboxes_scaled = postprocessed_outputs[0]['boxes'][keep]
    
    trainer = DetrTrainer(
        model=holy_detr,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=proc_datasets["train"],
        eval_dataset=proc_datasets["valid"],
        comput_metrics=compute_metrics,
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