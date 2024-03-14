# Which One Are You Referring To? Multimodal Object Identification in Situated Dialogue

Holy Lovenia, Samuel Cahyawijaya

-----

## Abstract

The demand for multimodal dialogue system has been rising in various domains, emphasizing the importance of interpreting multimodal inputs from conversational and situational context. One main challenge in multimodal dialogue understanding is multimodal object identification, which constitutes the ability to identify objects relevant to a multimodal user-system conversation. We explore three methods to tackle this problem and evaluate them on SIMMC 2.1. Our best method, scene-dialogue alignment, improves the performance by ~20% F1-score compared to the SIMMC 2.1 baselines. We provide analysis and discussion regarding the limitation of our methods and the potential directions for future works. 

## Setup
- Pull the latest `simmc2` and `detr` git submodules
- Install all required `python` dependency
```
pip install -r requirements.txt
```
- To obtain the preprocessed dialogue data, run `./run_data_preprocessing.sh`
- To obtain the preprocessed visual embeddings from DETR, run `./run_object_detection_finetuning.sh` and `./run_object_detection_prediction.sh`
- To obtain the preprocessed visual embeddings from ResNet-50 for the SIMMC 2.1 baselines, download from the [Google Drive](https://drive.google.com/file/d/1jr7r5Yaca80W5n0hizOakTG-F1ns6BGv/view?usp=sharing) provided by the organizers, then place the file under `./preprocessed_data/visual_features/`
- To obtain all fine-tuned models, download from [this OneDrive](https://hkustconnect-my.sharepoint.com/:f:/g/personal/hlovenia_connect_ust_hk/EnRdvg5F5TxIsM8yH3iG83cBHpkLz03n0SYKMrHQFEbAlA?e=nONyxZ), then place the folder at `./save`.

## File Structure
- git submodules
    - simmc2
    - detr
- fonts -> store the fonts for visualizing detection results
- model
    - clipper.py -> implementation of the CLIPPER model
    - holy_detr.py -> implementation of the SitCoM-DETR model
    - model_utils.py -> modeling utility functions
- trainer
    - detr_trainer.py -> a customized HuggingFace trainer class for DETR-derived model fine-tuning
- utils
    - args_helper.py -> argument parser used in the main script
    - data_utils.py -> data loading functions for various model fine-tuning formats (CLIP, DETR, SitCoM-DETR, etc)
    - utils.py -> common utility functions
- contextualized_object_detection_finetuning.py -> fine-tuning script for the SitCoM-DETR model
- contextualized_object_detection_prediction.py -> prediction script for the SitCoM-DETR model
- mdetr_zero_shot.py -> zero-shot prediction script for MDETR
- object_detection_finetuning.py -> fine-tuning script for the DETR-based model
- object_detection_prediction.py -> prediction script for the DETR-based model
- conv_image_alignment_finetuning.py -> fine-tuning script for the CLIP-based model using conversation data
- text_image_alignment_finetuning.py -> fine-tuning script for the CLIP-based model using scene-generated text data
- text_image_alignment_prediction.py -> prediction script for the CLIP-based model
- run_*.sh -> all the bash scripts to run the corresponding (*) experiment
- *.ipynb -> notebook files for sandbox and visualization

## Experiment Results
All experiment results are recorded on the following [Google Sheets](https://docs.google.com/spreadsheets/d/15QKo25eOP3GKPECHErpFg6pEEyB824fYTw24o8NplgM/edit?usp=sharing).

## LICENSE
The source code for the site is licensed under the MIT license, which you can find in the [LICENSE](LICENSE) file.
