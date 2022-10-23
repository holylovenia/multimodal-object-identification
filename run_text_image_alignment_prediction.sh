CUDA_VISIBLE_DEVICES=1 python3 text_image_alignment_prediction.py \
    --output_dir="./eval/text-im-align_pred" \
    --cache_dir="./cache/text-im-align" \
    --model_name_or_path="./save/conv-im-align/clipper/openai_clip-vit-base-patch32_linear_lr0.0001_bs256" \
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
    --utterance_turn both \
    --num_utterances 3 \
    --max_seq_length 77 
    # --prediction_path="./cache/text-im-align/_save_conv-im-align_openai_clip-vit-base-patch32_linear_lr00001_bs256_linear_lr5e-05_bs128/prediction_logits.pt"