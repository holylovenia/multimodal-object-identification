DATA_FOLDER="./simmc2/data/"
python simmc2/model/ambiguous_candidates/format_ambiguous_candidates_data.py \
	--simmc_train_json "./simmc2/data/simmc2.1_dials_dstc11_train.json" \
	--simmc_dev_json "./simmc2/data/simmc2.1_dials_dstc11_dev.json" \
	--simmc_devtest_json "./simmc2/data/simmc2.1_dials_dstc11_devtest.json" \
	--scene_json_folder "./simmc2/data/public/" \
	--ambiguous_candidates_save_path "./preprocessed_data/ambiguous_candidates/"