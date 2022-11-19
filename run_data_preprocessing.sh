DATA_FOLDER="/home/holy/datasets/simmc2.1/"

# python simmc2/model/ambiguous_candidates/format_ambiguous_candidates_data.py \
# 	--simmc_train_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_train.json" \
# 	--simmc_dev_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_dev.json" \
# 	--simmc_devtest_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_devtest.json" \
# 	--scene_json_folder "/home/holy/datasets/simmc2.1/public/" \
# 	--ambiguous_candidates_save_path "./preprocessed_data/ambiguous_candidates/"

# python simmc2/model/ambiguous_candidates/format_coref_candidates_data.py \
# 	--simmc_train_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_train.json" \
# 	--simmc_dev_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_dev.json" \
# 	--simmc_devtest_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_devtest.json" \
# 	--scene_json_folder "/home/holy/datasets/simmc2.1/public/" \
# 	--coref_candidates_save_path "./preprocessed_data/coref_candidates|no_mm/"

cd simmc2/model/mm_dst
./run_preprocess_gpt2.sh "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates|no_mm" /home/holy/datasets/simmc2.1
# ./run_preprocess_gpt2.sh "/home/holy/projects/ambiguous-mm-dialogue/preprocessed_data/coref_candidates|mm" /home/holy/datasets/simmc2.1