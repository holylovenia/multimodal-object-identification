DATA_FOLDER="/home/holy/datasets/simmc2.1/"
python simmc2/model/ambiguous_candidates/format_ambiguous_candidates_data.py \
	--simmc_train_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_train.json" \
	--simmc_dev_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_dev.json" \
	--simmc_devtest_json "/home/holy/datasets/simmc2.1/simmc2.1_dials_dstc11_devtest.json" \
	--scene_json_folder "/home/holy/datasets/simmc2.1/public/" \
	--ambiguous_candidates_save_path "./preprocessed_data/ambiguous_candidates/"