CUDA_VISIBLE_DEVICES=5 python3 text_image_alignment_prediction.py \
    --output_dir="./eval/text-im-align_pred" \
    --cache_dir="./cache/text-im-align" \
    --model_name_or_path="./save/conv-im-align/openai_clip-vit-base-patch32_linear_lr0.0001_bs128/checkpoint-8064" \
    --per_device_train_batch_size=64 \
    --per_device_eval_batch_size=64 \
    --gradient_accumulation_steps=2 \
    --dataloader_num_workers=32 \
    --remove_unused_columns=False \
    --report_to="tensorboard" \
    --seed=42 \
    --data_seed=42 \
    --dataloader_num_workers=8 \
    --overwrite_output_dir=True \
    --utterance_turn user \
    --num_utterances 1
    