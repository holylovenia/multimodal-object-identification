# DETR

# CUDA_VISIBLE_DEVICES=0 python train_model.py \
#     --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
#     --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
#     --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
#     --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/ambiguous-candidates/detr/gpt2" \
#     --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_detr.pt" \
#     --visual_feature_size 260 \
# 	--backbone gpt2 --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2


CUDA_VISIBLE_DEVICES=1 python train_model.py \
    --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/ambiguous-candidates/detr/bert" \
    --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_detr.pt" \
    --visual_feature_size 260 \
	--backbone bert --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2