#### MM-DST

cd simmc2/model/mm_dst
./run_train_gpt2.sh "/home/holy/projects/ambiguous-mm-dialogue/save/mm_dst|mm/mm_dst/"


# cd simmc2/model/ambiguous_candidates/
# pwd

#### COREF

# CUDA_VISIBLE_DEVICES=1 python train_model.py \
#     --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_train.json" \
#     --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_dev.json" \
#     --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_devtest.json" \
#     --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/mm-coref-candidates/amb-baseline/gpt2" \
#     --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_resnet50_simmc2.1.pt" \
#     --visual_feature_size 516 \
# 	--backbone gpt2 --use_gpu --num_epochs 10 --batch_size 16 --max_turns 3

# CUDA_VISIBLE_DEVICES=0 python train_model.py \
#     --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_train.json" \
#     --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_dev.json" \
#     --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates/simmc2.1_coref_candidates_dstc11_devtest.json" \
#     --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/mm-coref-candidates/amb-baseline/bert" \
#     --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_resnet50_simmc2.1.pt" \
#     --visual_feature_size 516 \
# 	--backbone bert --use_gpu --num_epochs 10 --batch_size 16 --max_turns 3

#### AMBIGUOUS

# CUDA_VISIBLE_DEVICES=0 python train_model.py \
#     --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
#     --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
#     --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
#     --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/ambiguous-candidates/detr/gpt2" \
#     --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_detr.pt" \
#     --visual_feature_size 260 \
# 	--backbone gpt2 --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2

# CUDA_VISIBLE_DEVICES=1 python train_model.py \
#     --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
#     --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
#     --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
#     --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/ambiguous-candidates/detr/bert" \
#     --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_detr.pt" \
#     --visual_feature_size 260 \
# 	--backbone bert --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2