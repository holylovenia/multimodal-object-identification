# Multimodal Object Identification in Situated Dialogue

## Setup
- Pull the latest `simmc2` and `detr` git submodules
- Install all required `python` dependency
```
pip install -r requirements.txt
```


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
- object_detection_finetuning.py -> fine-tuning script for the DETR-based model
- object_detection_prediction.py -> prediction script for the DETR-based model
- conv_image_alignment_finetuning.py -> fine-tuning script for the CLIP-based model using conversation data
- text_image_alignment_finetuning.py -> fine-tuning script for the CLIP-based model using scene-generated text data
- text_image_alignment_prediction.py -> prediction script for the CLIP-based model
- run_*.sh -> all the bash scripts to run the corresponding (*) experiment
- *.ipynb -> notebook files for sandbox and visualization

## Experiment Results
All experiment results are recorded on the following [Google Sheet](https://docs.google.com/spreadsheets/d/15QKo25eOP3GKPECHErpFg6pEEyB824fYTw24o8NplgM/edit?usp=sharing)

## LICENSE
The source code for the site is licensed under the MIT license, which you can find in the [LICENSE](LICENSE) file.
