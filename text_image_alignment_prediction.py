from PIL import ImageFile
from copy import deepcopy
from datasets import load_from_disk, set_caching_enabled
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
from transformers import HfArgumentParser
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

from simmc2.model.utils import ambiguous_candidates_evaluation as eval_utils
from trainer.detr_trainer import DetrTrainer 
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
    eval_dset, gold_data = data_utils.load_image_text_eval_dataset()
    eval_dset = eval_dset.map(
        data_utils.convert_dialogue_to_caption,
        num_proc=data_args.preprocessing_num_workers,
        desc="convert object attributes to caption",
        load_from_cache_file=True,
        cache_file_name=os.path.join(cache_dir_path, "ds_converted.arrow"),
        fn_kwargs={"num_utterances": data_args.num_utterances},
        remove_columns=["dialogue"]
    )

    # Preprocessing
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(model_args.model_name_or_path)
    processor = transformers.VisionTextDualEncoderProcessor(feature_extractor, tokenizer)
    
    eval_dset = eval_dset.map(
        data_utils.tokenize_captions,
        num_proc=data_args.preprocessing_num_workers,
        desc="tokenize captions",
        fn_kwargs={
            "tokenizer": tokenizer,
            "max_seq_length": data_args.max_seq_length,
        }
    )

    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    eval_transforms = Compose(
        [
            Resize(feature_extractor.size),
            CenterCrop(feature_extractor.size),
            ToTensor(),
            normalize,
        ]
    )

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

    eval_dset = eval_dset.with_transform(eval_image_preprocess)

    # Training and evaluation
    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        input_ids = torch.tensor([example["input_ids"] for example in examples], dtype=torch.long)
        attention_mask = torch.tensor([example["attention_mask"] for example in examples], dtype=torch.long)
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_loss": True,
    }

    model = transformers.VisionTextDualEncoderModel.from_pretrained(model_args.model_name_or_path)

    def compute_metrics(p):
        """Aggregate predictions & compute evaluation metric per utterance"""
        def sigmoid_array(x):                                        
            return 1 / (1 + np.exp(-x))
        preds = sigmoid_array(p.predictions, axis=1) # Assume this is logits_per_image
        
        pred_dict = {'dialog_id': [], 'turn_id': [], 'pred': []}
        for row, pred in zip(eval_dset, preds):
            pred_dict['dialog_id'] = row.dialog_id
            pred_dict['turn_id'] = row.turn_id
            pred_dict['pred'] = prediction
            
        df = pd.DataFrame(pred_dict)
        agg_preds = df.groupby(['dialog_id','turn_id'])['pred'].apply(list).reset_index().to_dict(orient='records')
        
        results = collections.defaultdict(list)
        for agg_pred in agg_preds:
            dialog_id, turn_id, preds = agg_pred['dialog_id'], agg_pred['turn_id'], agg_pred['pred']                
            new_instance = {
                "turn_id": turn_id,
                "disambiguation_candidates": preds
            }
            results[dialog_id].append(new_instance)

        # Restructure results JSON and save.
        results = [{
            "dialog_id": dialog_id,
            "predictions": predictions,
        } for dialog_id, predictions in results.items()]
        
        metrics = eval_utils.evaluate_ambiguous_candidates(gold_data, results)
        
        print('Eval Metrics', metrics["recall"], metrics["precision"], metrics["f1"])
        return metrics
    
    trainer = DetrTrainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=processor,
        compute_metrics=compute_metrics
    )

    # Evaluation
    metrics = trainer.evaluate(eval_dset)
    print('Metrics', metrics)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)    

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
                f"Output directory ({training_args.output_dir}) already exists and is not empty."
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