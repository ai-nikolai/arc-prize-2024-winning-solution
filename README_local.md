# Readme to get going locally

## Installation (tested on 28.01.2025)
1. python 3.10
2. Requirements (preferred in virtualenv)
```bash
pip3 install -r requirements.txt
```
3. Run the notebook `local_notebooks/arc-prize-2024_main.ipynb`
- Note augmentation is off for training and evaluation (and evaluation is set to the eval set.)

## Result you should be getting using the default settings as above (i.e. augmentation off for training and testing, running eval on the eval dataset and using a single GPU - 24GB)

1. Computation Time - RTX 3090
- Training: ~6m
- Inference: ~67m
- Selection: ~