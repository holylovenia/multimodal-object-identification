cd simmc2/model/ambiguous_candidates/

pwd

echo "=========================="
echo "Heuristic Baseline: Random"
CUDA_VISIBLE_DEVICES=0 python evaluate_heuristic_baselines.py \
    --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --baseline_type random

echo "=========================="
echo "Heuristic Baseline: All Objects"
CUDA_VISIBLE_DEVICES=0 python evaluate_heuristic_baselines.py \
    --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --baseline_type all_objects

echo "=========================="
echo "Heuristic Baseline: No Objects"
CUDA_VISIBLE_DEVICES=0 python evaluate_heuristic_baselines.py \
    --train_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_train.json" \
    --dev_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_dev.json" \
    --devtest_file "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json" \
    --baseline_type no_objects