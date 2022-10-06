cd simmc2/model/ambiguous_candidates/

pwd

CUDA_VISIBLE_DEVICES=0 python train_model.py \
    --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --result_save_path "/home/holy/projects/ambiguous-mm-dialogue/results/" \
    --visual_feature_path "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/visual_features/visual_features_resnet50_simmc2.1.pt" \
    --visual_feature_size 516 \
	--backbone bert --use_gpu --num_epochs 10 --batch_size 16 --max_turns 2