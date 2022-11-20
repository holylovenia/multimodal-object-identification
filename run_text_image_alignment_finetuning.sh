# CUDA_VISIBLE_DEVICES=0 python text_image_alignment_finetuning.py \
#     --output_dir="./save/coref_candidates|mm/text-im-align/clip" \
#     --cache_dir="./cache/coref_candidates|mm/text-im-align/clip" \
#     --model_name_or_path="openai/clip-vit-base-patch32" \
#     --train_dataset_path "./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_train.json" \
#     --dev_dataset_path "./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_dev.json" \
#     --devtest_dataset_path "./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_devtest.json" \
#     --additional_special_token_path "./preprocessed_data/coref_candidates|mm/simmc2_special_tokens.json" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --num_train_epochs=200 \
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

CUDA_VISIBLE_DEVICES=3 python text_image_alignment_finetuning.py \
    --output_dir="./save/coref_candidates|no_mm/text-im-align/clip" \
    --cache_dir="./cache/coref_candidates|no_mm/text-im-align/clip" \
    --model_name_or_path="openai/clip-vit-base-patch32" \
    --train_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_train.json" \
    --dev_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_dev.json" \
    --devtest_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_devtest.json" \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --num_train_epochs=200 \
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

# CUDA_VISIBLE_DEVICES=1 python text_image_alignment_finetuning.py \
#     --output_dir="./save/ambiguous_candidates/text-im-align/clip" \
#     --cache_dir="./cache/ambiguous_candidates/text-im-align/clip" \
#     --model_name_or_path="openai/clip-vit-base-patch32" \
#     --train_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
#     --dev_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
#     --devtest_dataset_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --num_train_epochs=200 \
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