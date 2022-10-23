CUDA_VISIBLE_DEVICES=0 python conv_image_alignment_finetuning.py \
    --output_dir="./save/conv-im-align/clipper" \
    --cache_dir="./cache/conv-im-align/clipper" \
    --model_name_or_path="openai/clip-vit-base-patch32" \
    --vision_model_name_or_path="openai/clip-vit-base-patch32" \
    --text_model_name_or_path="roberta-base" \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs=200 \
    --max_seq_length=77 \
    --fp16=True \
    --save_strategy="epoch" \
    --save_steps=1 \
    --logging_strategy="epoch" \
    --logging_steps=1 \
    --evaluation_strategy="epoch" \
    --eval_steps=1 \
    --gradient_accumulation_steps=8 \
    --eval_accumulation_steps=8 \
    --learning_rate=1e-4 \
    --save_total_limit=1 \
    --remove_unused_columns=False \
    --report_to="tensorboard" \
    --seed=42 \
    --data_seed=42 \
    --load_best_model_at_end=True \
    --dataloader_num_workers=4 \
    --overwrite_output_dir=True

# # DUMMY FOR TESTING
# CUDA_VISIBLE_DEVICES=0 python conv_image_alignment_finetuning.py \
#     --output_dir="./save/test" \
#     --cache_dir="./cache/test" \
#     --model_name_or_path="openai/clip-vit-base-patch32" \
#     --vision_model_name_or_path="openai/clip-vit-base-patch32" \
#     --text_model_name_or_path="roberta-base" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --num_train_epochs=10 \
#     --fp16=True \
#     --save_strategy="epoch" \
#     --save_steps=1 \
#     --logging_strategy="epoch" \
#     --logging_steps=1 \
#     --evaluation_strategy="epoch" \
#     --eval_steps=1 \
#     --gradient_accumulation_steps=8 \
#     --eval_accumulation_steps=8 \
#     --learning_rate=1e-4 \
#     --save_total_limit=1 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --load_best_model_at_end=True \
#    --dataloader_num_workers=4 \
#     --overwrite_output_dir=True