from PIL import ImageFile
from copy import deepcopy
from datasets import load_from_disk, set_caching_enabled
from detr import CocoEvaluator
from model import model_utils
from model.clipper import CLIPPERModel
from utils import data_utils, utils
from utils.args_helper import (
    DataArguments,
    ModelArguments,
    TrainingArguments
)
from tqdm import tqdm
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomVerticalFlip,
    RandomResizedCrop,
    RandomRotation,
    Resize,
    ToTensor,
)
from trainer.detr_trainer import DetrTrainer
from transformers import HfArgumentParser, CLIPModel
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
    training_args.output_dir="{}/{}_{}_lr{}_bs{}".format(
        training_args.output_dir,
        model_args.model_name_or_path.replace("/", "_"),
        training_args.lr_scheduler_type,
        training_args.learning_rate,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    os.makedirs(training_args.output_dir, exist_ok=True)
    cache_dir_path = "{}/{}_{}_lr{}_bs{}".format(
        data_args.cache_dir_name,
        model_args.model_name_or_path.replace("/", "_"),
        training_args.lr_scheduler_type,
        training_args.learning_rate,
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    )
    os.makedirs(cache_dir_path, exist_ok=True)

    # Data loading
    raw_datasets = datasets.DatasetDict()
    raw_datasets["train"] = data_utils.load_image_conv_dataset(
        data_path=data_args.train_dataset_path,
        return_gt_labels=False)
    raw_datasets["dev"] = data_utils.load_image_conv_dataset(
        data_path=data_args.dev_dataset_path,
        return_gt_labels=False)
    raw_datasets["devtest"] = data_utils.load_image_conv_dataset(
        data_path=data_args.devtest_dataset_path,
        return_gt_labels=False)

    raw_datasets = raw_datasets.map(
        data_utils.convert_dialogue_to_caption,
        num_proc=data_args.preprocessing_num_workers,
        desc="convert dialogue to caption",
        load_from_cache_file=True,
        cache_file_names={
            "train": os.path.join(cache_dir_path, "train_converted.arrow"),
            "dev": os.path.join(cache_dir_path, "dev_converted.arrow"),
            "devtest": os.path.join(cache_dir_path, "devtest_converted.arrow"),
        }
    )

    # Preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    tokenizer.add_special_tokens({"additional_special_tokens": ["<USER>", "<SYS>"]})
    if data_args.additional_special_token_path is not None and os.path.isfile(data_args.additional_special_token_path):
        with open(data_args.additional_special_token_path, "rb") as handle:
            special_tokens_dict = json.load(handle)
            num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        logger.info(f"Added {num_added_toks} tokens")
        logger.info(f"All special tokens: {tokenizer.all_special_tokens}")
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    processor = transformers.AutoProcessor.from_pretrained(model_args.model_name_or_path)

    proc_datasets = raw_datasets.map(
        data_utils.tokenize_captions,
        num_proc=data_args.preprocessing_num_workers,
        desc="tokenize captions",
        load_from_cache_file=True,
        cache_file_names={
            "train": os.path.join(cache_dir_path, "train_tokenized.arrow"),
            "dev": os.path.join(cache_dir_path, "dev_tokenized.arrow"),
            "devtest": os.path.join(cache_dir_path, "devtest_tokenized.arrow"),
        },
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": data_args.max_seq_length,
        }
    )

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    train_transforms = Compose(
            [
                Resize(feature_extractor.size),
                # CenterCrop(feature_extractor.size),
                # RandomHorizontalFlip(),
                # RandomVerticalFlip(),
                # RandomRotation(5),            
                ToTensor(),
                # normalize,
            ]
        )

    eval_transforms = Compose(
            [
                Resize(feature_extractor.size),
                # CenterCrop(feature_extractor.size),
                ToTensor(),
                # normalize,
            ]
        )

    def train_image_preprocess(example_batch):
        images = [
            train_transforms(
                image.convert("RGB").crop((
                    bbox[0], bbox[1], bbox[0]+max(5, bbox[3]), bbox[1]+max(5, bbox[2])
                ))
            )
            for image, bbox in zip(example_batch["image"], example_batch["bbox"])
        ]
        captions = [caption for caption in example_batch["caption"]]
        example_batch["pixel_values"] = feature_extractor(
            images=images, text=captions, return_tensors="pt")["pixel_values"]
        return example_batch

    def eval_image_preprocess(example_batch):
        images = [
            eval_transforms(
                image.convert("RGB").crop((
                    bbox[0], bbox[1], bbox[0]+max(5, bbox[3]), bbox[1]+max(5, bbox[2])
                ))
            )
            for image, bbox in zip(example_batch["image"], example_batch["bbox"])
        ]
        captions = [caption for caption in example_batch["caption"]]
        example_batch["pixel_values"] = feature_extractor(
            images=images, text=captions, return_tensors="pt")["pixel_values"]
        return example_batch

    proc_datasets["train"] = proc_datasets["train"].with_transform(train_image_preprocess)
    proc_datasets["dev"] = proc_datasets["dev"].with_transform(eval_image_preprocess)
    proc_datasets["devtest"] = proc_datasets["devtest"].with_transform(eval_image_preprocess)

    # Training and evaluation
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        
        if model_args.include_other_similar_objects or model_args.include_other_referred_objects:
            object_ids = torch.tensor([example["object_id"] for example in examples])
            prefab_object_ids = torch.tensor([example["prefab_object_id"] for example in examples])
            other_ambig_object_unique_ids = [example["other_ambig_object_unique_ids"] for example in examples]
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_loss": True,
                "object_ids": object_ids,
                "prefab_object_ids": prefab_object_ids,
                "other_ambig_object_unique_ids": other_ambig_object_unique_ids,
            }
        else:
            return {
                "pixel_values": pixel_values,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "return_loss": True,
            }

    # config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path)
    # if data_args.max_seq_length > 77: # CLIP's default absolute max position embeddings
    #     config.update({"max_position_embeddings": data_args.max_seq_length})
    if model_args.include_other_similar_objects is False and model_args.include_other_referred_objects is False:
        print("CLIP")
        model = CLIPModel.from_pretrained(model_args.model_name_or_path)
    else:
        print("CLIPPER", model_args.include_other_similar_objects, model_args.include_other_referred_objects)
        model = CLIPPERModel.from_pretrained(model_args.model_name_or_path)
        model.modify_learning_objective(model_args)
    # if data_args.max_seq_length > 77: # CLIP's default absolute max position embeddings
    #     model = model_utils._resize_position_embeddings(model, data_args.max_seq_length)
    # print(model.vision_model.embeddings.position_embedding, model.text_model.embeddings.position_embedding)
    
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=proc_datasets["train"],
        eval_dataset=proc_datasets["dev"],
        tokenizer=processor,
        callbacks=[transformers.EarlyStoppingCallback(early_stopping_patience=10)],
    )

    # Training
    train_results = trainer.train()
    trainer.save_model()

    # Evaluation
    metrics = trainer.evaluate(proc_datasets["dev"])
    trainer.log_metrics("dev", metrics)
    trainer.save_metrics("dev", metrics)

    metrics = trainer.evaluate(proc_datasets["devtest"])
    trainer.log_metrics("devtest", metrics)
    trainer.save_metrics("devtest", metrics)
    

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
            "./log/log_{}_{}_lr{}_bs{}".format(
                model_args.model_name_or_path.replace("/", "_"),
                training_args.lr_scheduler_type,
                training_args.learning_rate,
                training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
            ), mode="w")],
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