CUDA_VISIBLE_DEVICES=3 python3 contextualized_object_detection_prediction.py \
    --output_dir="./eval_holy/obj-det" --cache_dir="./cache_holy/obj-det" \
    --checkpoint_path="./save_holy/obj-det/facebook/detr-resnet-50/checkpoint-19040" \
    --model_name_or_path="facebook/detr-resnet-50" --text_model_name_or_path="roberta-base" \
    --train_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json' \
    --dev_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json' \
    --devtest_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json' \
    --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --num_train_epochs=200 --fp16=True \
    --save_strategy="epoch" --save_steps=1 --save_total_limit=1 --load_best_model_at_end=True \
    --logging_strategy="epoch" --logging_steps=1 --report_to="tensorboard" \
    --evaluation_strategy="epoch" --eval_steps=1 --eval_accumulation_steps=8 \
    --seed=42 --data_seed=42 --dataloader_num_workers=4 \
    --gradient_accumulation_steps=4 --learning_rate=1e-4 --remove_unused_columns=False \
    --overwrite_output_dir=True
    
CUDA_VISIBLE_DEVICES=1 python3 contextualized_object_detection_finetuning.py \
    --output_dir="./save_holy/obj-det" --cache_dir="./cache_holy/obj-det" \
    --model_name_or_path="facebook/detr-resnet-50" --text_model_name_or_path="roberta-base" \
    --train_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json' \
    --dev_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json' \
    --devtest_dataset_path='./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json' \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --num_train_epochs=200 --fp16=True \
    --save_strategy="epoch" --save_steps=1 --save_total_limit=1 --load_best_model_at_end=True \
    --logging_strategy="epoch" --logging_steps=1 --report_to="tensorboard" \
    --evaluation_strategy="epoch" --eval_steps=1 --eval_accumulation_steps=8 \
    --seed=42 --data_seed=42 --dataloader_num_workers=16 \
    --gradient_accumulation_steps=4 --learning_rate=1e-4 --remove_unused_columns=False \
    --overwrite_output_dir=True --augment_with_scene_data 2>&1 | tee sitcom_detr_roberta_yes_augment.log