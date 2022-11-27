#### COREF

# CUDA_VISIBLE_DEVICES=0 python3 text_image_alignment_prediction.py \
#     --output_dir="./eval/ambiguous_candidates/conv-im-align/off_the_shelf_clip" \
#     --cache_dir="./cache/ambiguous_candidates/conv-im-align/off_the_shelf_clip" \
#     --model_name_or_path="openai/clip-vit-base-patch32" \
#     --train_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
#     --dev_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
#     --devtest_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --gradient_accumulation_steps=2 \
#     --max_seq_length=77 \
#     --dataloader_num_workers=32 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --dataloader_num_workers=8 \
#     --overwrite_output_dir=True \
#     --utterance_turn both \
#     --num_utterances 3 \
#     --prediction_path ./cache/ambiguous_candidates/conv-im-align/off_the_shelf_clip/openai_clip-vit-base-patch32_linear_lr5e-05_bs64/prediction_logits.pt


CUDA_VISIBLE_DEVICES=1 python3 text_image_alignment_prediction.py \
    --output_dir="./eval/ambiguous_candidates/text-im-align/clip" \
    --cache_dir="./cache/ambiguous_candidates/text-im-align/clip" \
    --model_name_or_path="./save/ambiguous_candidates/text-im-align/clip/openai_clip-vit-base-patch32_linear_lr1e-05_bs32" \
    --train_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_dataset_path="./preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --per_device_train_batch_size=32 \
    --per_device_eval_batch_size=32 \
    --gradient_accumulation_steps=2 \
    --max_seq_length=77 \
    --dataloader_num_workers=32 \
    --remove_unused_columns=False \
    --report_to="tensorboard" \
    --seed=42 \
    --data_seed=42 \
    --dataloader_num_workers=8 \
    --overwrite_output_dir=True \
    --utterance_turn both \
    --num_utterances 3 \
    --prediction_path ./cache/ambiguous_candidates/text-im-align/clip/_save_ambiguous_candidates_text-im-align_clip_openai_clip-vit-base-patch32_linear_lr1e-05_bs32_linear_lr5e-05_bs64/prediction_logits.pt


# CUDA_VISIBLE_DEVICES=0 python3 text_image_alignment_prediction.py \
#     --output_dir="./eval/coref_candidates|mm/conv-im-align/clipper_v3" \
#     --cache_dir="./cache/coref_candidates|mm/conv-im-align/clipper_v3" \
#     --model_name_or_path="./save/coref_candidates|mm/conv-im-align/clipper_v3/openai_clip-vit-base-patch32_linear_lr1e-5_bs256" \
#     --train_dataset_path="./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_train.json" \
#     --dev_dataset_path="./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_dev.json" \
#     --devtest_dataset_path="./preprocessed_data/coref_candidates|mm/simmc2.1_coref_candidates_dstc11_devtest.json" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --gradient_accumulation_steps=2 \
#     --max_seq_length=77 \
#     --dataloader_num_workers=32 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --dataloader_num_workers=8 \
#     --overwrite_output_dir=True \
#     --utterance_turn both \
#     --num_utterances 3 \
#     --prediction_path "/home/holy/projects/ambiguous-mm-dialogue/cache/coref_candidates|mm/conv-im-align/clipper_v3/_save_coref_candidates|mm_conv-im-align_clipper_v3_openai_clip-vit-base-patch32_linear_lr00001_bs256_linear_lr5e-05_bs64/prediction_logits.pt"



# CUDA_VISIBLE_DEVICES=0 python3 text_image_alignment_prediction.py \
#     --output_dir="./eval/coref_candidates|no_mm/conv-im-align/clipper_v1" \
#     --cache_dir="./cache/coref_candidates|no_mm/conv-im-align/clipper_v1" \
#     --model_name_or_path="./save/coref_candidates|no_mm/conv-im-align/clipper_v1/openai_clip-vit-base-patch32_linear_lr0.0001_bs256" \
#     --train_dataset_path="./preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_train.json" \
#     --dev_dataset_path="./preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_dev.json" \
#     --devtest_dataset_path="./preprocessed_data/coref_candidates|no_mm/simmc2.1_coref_candidates_dstc11_devtest.json" \
#     --per_device_train_batch_size=32 \
#     --per_device_eval_batch_size=32 \
#     --gradient_accumulation_steps=2 \
#     --max_seq_length=77 \
#     --dataloader_num_workers=32 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --dataloader_num_workers=8 \
#     --overwrite_output_dir=True \
#     --utterance_turn both \
#     --num_utterances 3 \
#     --prediction_path "/home/holy/projects/ambiguous-mm-dialogue/cache/coref_candidates|no_mm/conv-im-align/clipper_v1/_save_coref_candidates|no_mm_conv-im-align_clipper_v1_openai_clip-vit-base-patch32_linear_lr00001_bs256_linear_lr5e-05_bs64/prediction_logits.pt"


#### AMBIGUOUS

# CUDA_VISIBLE_DEVICES=0 python3 text_image_alignment_prediction.py \
#     --output_dir="./eval/conv-im-align/clipper_neighbors" \
#     --cache_dir="./cache/conv-im-align/clipper_neighbors" \
#     --model_name_or_path="./save/conv-im-align/clipper_neighbors/openai_clip-vit-base-patch32_linear_lr0.0001_bs256" \
#     --per_device_train_batch_size=64 \
#     --per_device_eval_batch_size=64 \
#     --gradient_accumulation_steps=2 \
#     --max_seq_length=77 \
#     --dataloader_num_workers=32 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --dataloader_num_workers=8 \
#     --overwrite_output_dir=True \
#     --utterance_turn both \
#     --num_utterances 3 \
#     --prediction_path ./cache/conv-im-align/clipper_neighbors/_save_conv-im-align_clipper_neighbors_openai_clip-vit-base-patch32_linear_lr00001_bs256_linear_lr5e-05_bs128/prediction_logits.pt

# CUDA_VISIBLE_DEVICES=1 python3 text_image_alignment_prediction.py \
#     --output_dir="./eval/conv-im-align/clip_sigmoid" \
#     --cache_dir="./cache/conv-im-align/clip_sigmoid" \
#     --model_name_or_path="./save/conv-im-align/clip_sigmoid/openai_clip-vit-base-patch32_linear_lr0.0001_bs256" \
#     --per_device_train_batch_size=64 \
#     --per_device_eval_batch_size=64 \
#     --gradient_accumulation_steps=2 \
#     --max_seq_length=77 \
#     --dataloader_num_workers=32 \
#     --remove_unused_columns=False \
#     --report_to="tensorboard" \
#     --seed=42 \
#     --data_seed=42 \
#     --dataloader_num_workers=8 \
#     --overwrite_output_dir=True \
#     --utterance_turn both \
#     --num_utterances 3 \
#     --prediction_path ./cache/conv-im-align/clip_sigmoid/_save_conv-im-align_clip_sigmoid_openai_clip-vit-base-patch32_linear_lr00001_bs256_linear_lr5e-05_bs128/prediction_logits.pt