# Original Model Notebooks

This folder contains the original Jupyter notebooks for individual model training.

## Contents

- `Model_cnn_one_block.ipynb` - 1-block CNN models (base, dropout, batchnorm)
- `Model_cnn_two_block.ipynb` - 2-block CNN models (base, dropout, batchnorm)
- `Model_cnn_three_block.ipynb` - 3-block CNN models (base, dropout, batchnorm)
- `Model_cnn_four_block.ipynb` - 4-block CNN models (base, dropout, batchnorm)
- `Model_cnn_five_block.ipynb` - 5-block CNN models (base, dropout, batchnorm)
- `Model_cnn_six_block.ipynb` - 6-block CNN models (base, dropout, batchnorm)

## Note

These are the original notebooks used during development. For training all models at once, use:

- **`All_Models_Training.ipynb`** (in the root directory) - Trains all 18 models
- **`train_test.py`** - Python script for training individual models
- **`main.py`** - Main pipeline script

The models defined in these notebooks have been consolidated into `models.py` for easier reuse.

