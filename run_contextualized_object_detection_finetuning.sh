CUDA_VISIBLE_DEVICES=1 python3 contextualized_object_detection_finetuning.py \
    --output_dir="./save_holy/obj-det" --cache_dir="./cache_holy/obj-det" \
    --model_name_or_path="facebook/detr-resnet-50" --text_model_name_or_path="roberta-base" \
    --train_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json' \
    --dev_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json' \
    --devtest_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json' \
    --per_device_train_batch_size=32 --per_device_eval_batch_size=32 --num_train_epochs=200 --fp16=True \
    --save_strategy="epoch" --save_steps=1 --save_total_limit=1 --load_best_model_at_end=True \
    --logging_strategy="epoch" --logging_steps=1 --report_to="tensorboard" \
    --evaluation_strategy="epoch" --eval_steps=1 --eval_accumulation_steps=32 \
    --seed=42 --data_seed=42 --dataloader_num_workers=32 \
    --gradient_accumulation_steps=2 --learning_rate=1e-4 --remove_unused_columns=False \
    --overwrite_output_dir=True
    
CUDA_VISIBLE_DEVICES=1 python3 contextualized_object_detection_finetuning.py \
    --output_dir="./save_holy/obj-det" --cache_dir="./cache_holy/obj-det" \
    --model_name_or_path="facebook/detr-resnet-50" --text_model_name_or_path="roberta-base" \
    --train_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json' \
    --dev_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json' \
    --devtest_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json' \
    --per_device_train_batch_size=32 --per_device_eval_batch_size=32 --num_train_epochs=200 --fp16=True \
    --save_strategy="epoch" --save_steps=1 --save_total_limit=1 --load_best_model_at_end=True \
    --logging_strategy="epoch" --logging_steps=1 --report_to="tensorboard" \
    --evaluation_strategy="epoch" --eval_steps=1 --eval_accumulation_steps=32 \
    --seed=42 --data_seed=42 --dataloader_num_workers=32 \
    --gradient_accumulation_steps=2 --learning_rate=1e-4 --remove_unused_columns=False \
    --overwrite_output_dir=True