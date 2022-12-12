from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default="gpt2", metadata={"help": "The path of the HuggingFace model."}
    )    
    text_model_name_or_path: Optional[str] = field(
        default="bert-base-uncased", metadata={"help": "The path of the HuggingFace text model for language model."}
    )
    vision_model_name_or_path: Optional[str] = field(
        default="facebook/detr-resnet-50", metadata={"help": "The path of the HuggingFace vision model for object detection."}
    )
    checkpoint_path: Optional[str] = field(
        default=None, metadata={"help": "The path of the checkpoint of SitCoM-DETR."}
    )
    include_other_similar_objects: Optional[bool] = field(
        default=False, metadata={"help": "Whether to include other similar objects or not"}
    )
    include_other_referred_objects: Optional[bool] = field(
        default=False, metadata={"help": "Whether to include other referred objects or not"}
    )
    positive_weight: Optional[float] = field(
        default=1.0,
        metadata={"help": "Weight for positive examples for class imbalance"},
    )
    

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    train_dataset_path: str = field(
        metadata={"help": "Train dataset path"}
    )
    dev_dataset_path: str = field(
        metadata={"help": "Dev dataset path"}
    )
    devtest_dataset_path: str = field(
        metadata={"help": "Devtest dataset path"}
    )
    augment_with_scene_data:  Optional[bool] = field(
        default=False, metadata={"help": "Whether to include train dataset from scenery or not"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    cache_dir_name: Optional[str] = field(
        default="cache",
        metadata={"help": "Name of cache directory"},
    )
    max_seq_length: Optional[int] = field(
        default=36,
        metadata={"help": "Maximum sequence length"}
    )
    preprocessed_data_dir: Optional[str] = field(
        default="./preprocessed_data"
    )
    num_utterances: Optional[int] = field(
        default=3
    )
    utterance_turn: Optional[str] = field(
        default='both'
    )
    max_turns: Optional[int] = field(
        default = 5,
    )
    prediction_path: Optional[str] = field(
        default=None
    )
    additional_special_token_path: Optional[str] = field(
        default=None
    )

@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertraining to the training pipeline.
    """
    output_dir: Optional[str] = field(
        default="./save",
        metadata={"help": "Output directory"},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=1,
        metadata={"help": "Evaluation accumulation steps"}
    )
    adam_epsilon: Optional[float] = field(
        default=1e-8,
        metadata={"help": "Eps for Adam optimizer"},
    )