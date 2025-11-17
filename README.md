# Cats vs Dogs Classification with CNN

A machine learning project for binary image classification of cats and dogs using Convolutional Neural Networks (CNNs). This repository contains optimized Python scripts and Jupyter notebooks for data preparation, model training, and evaluation.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Main Pipeline Script](#main-pipeline-script)
  - [Data Preparation](#data-preparation)
  - [Model Training](#model-training)
  - [Jupyter Notebooks](#jupyter-notebooks)
- [Models](#models)
- [Results](#results)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Dataset](#dataset)
- [Author & Acknowledgments](#author--acknowledgments)

## 🎯 Overview

This project implements various CNN architectures to classify images of cats and dogs. It includes:

- **6 different model architectures** (1-block to 6-block CNNs)
- **3 variants per architecture** (base, dropout, batch normalization)
- **Total: 18 model configurations**
- **Automated data pipeline** (download, preprocessing, training)
- **Cross-platform support** (Windows, Mac, Linux, Google Colab)

## ✨ Features

- 🔄 **End-to-end pipeline**: From data download to model evaluation
- 🎛️ **Flexible architecture**: 18 different model configurations
- 📊 **Comprehensive logging**: Track training progress and metrics
- 💾 **Model persistence**: Save trained models and training history
- 🖥️ **Cross-platform**: Works on local machines and Google Colab
- 📈 **Visualization**: Automatic generation of training plots
- 📓 **Jupyter notebooks**: Interactive exploration and training

## 📁 Project Structure

```
.
├── main.py                      # Main pipeline orchestration script
├── data_preparation.py          # Data download and preprocessing
├── models.py                    # CNN model definitions (18 models)
├── train_test.py                # Model training and evaluation
├── requirements.txt             # Python dependencies
├── README.md                    # This comprehensive guide
├── All_Models_Training.ipynb    # Jupyter notebook to train all models
├── Data_Preparation.ipynb       # Data preparation notebook
├── notebooks/                   # Original model notebooks
│   ├── Model_cnn_one_block.ipynb
│   ├── Model_cnn_two_block.ipynb
│   ├── Model_cnn_three_block.ipynb
│   ├── Model_cnn_four_block.ipynb
│   ├── Model_cnn_five_block.ipynb
│   └── Model_cnn_six_block.ipynb
└── CatsDogs/                    # Dataset directory (created after data prep)
    ├── Cats/                    # Cat images
    ├── Dogs/                    # Dog images
    └── Pickles/                 # Preprocessed data
        ├── X.pickle             # Image arrays
        └── y.pickle             # Labels
```

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Setup

1. **Clone the repository**:
```bash
git clone <repository-url>
cd MachineLearning-CesaBianchi-CatsDogs
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **(Optional) Setup Kaggle API** for dataset download:
   
   **Quick Setup:**
   - Go to [Kaggle Account Settings](https://www.kaggle.com/account)
   - Scroll down to the "API" section
   - Click "Create New API Token" to download `kaggle.json`
   - Run: `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   
   **Detailed Setup:**
   
   **Step 1: Get Your Kaggle API Token**
   1. Go to **https://www.kaggle.com/account**
   2. Scroll down to the **"API"** section
   3. Click **"Create New API Token"**
   4. This will download a file named `kaggle.json` to your Downloads folder
   
   **Step 2: Place the Credentials File**
   
   **On Mac/Linux:**
   ```bash
   # Create the .kaggle directory if it doesn't exist
   mkdir -p ~/.kaggle
   
   # Move the downloaded kaggle.json file
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set proper permissions (important!)
   chmod 600 ~/.kaggle/kaggle.json
   ```
   
   **On Windows:**
   ```powershell
   # Create the .kaggle directory
   mkdir $env:USERPROFILE\.kaggle
   
   # Move the downloaded kaggle.json file
   move $env:USERPROFILE\Downloads\kaggle.json $env:USERPROFILE\.kaggle\
   ```
   
   **Step 3: Verify Setup**
   ```bash
   python -c "import kaggle; print('✓ Kaggle API is ready!')"
   ```
   
   **Alternative: Manual Setup**
   
   If you prefer to create the file manually:
   1. Create the directory: `~/.kaggle/` (or `C:\Users\<username>\.kaggle\` on Windows)
   2. Create a file named `kaggle.json` in that directory
   3. Add your credentials in this format:
   ```json
   {
     "username": "your_kaggle_username",
     "key": "your_api_key_here"
   }
   ```
   
   **Using Setup Scripts:**
   - Run `python setup_kaggle.py` for interactive setup
   - Or run `./setup_kaggle_quick.sh` after downloading kaggle.json

## 🏃 Quick Start

### Option 1: Full Pipeline (Recommended)

Run the complete pipeline from data preparation to training:

```bash
python main.py --prepare-data --download --train --blocks 3 --variant batchnorm --save-model --save-plot

# Or use Google Drive pickles (no Kaggle API needed)
python data_preparation.py --download-pickles
python train_test.py --blocks 3 --variant batchnorm --epochs 50 --save-model --save-plot
```

### Option 2: Step by Step

1. **Prepare the data** (choose one method):
```bash
# Option A: Download from Kaggle (requires Kaggle API setup)
python data_preparation.py --download

# Option B: Download pre-prepared pickles from Google Drive (recommended, no API needed)
python data_preparation.py --download-pickles
```

2. **Train a model**:
```bash
python train_test.py --blocks 3 --variant batchnorm --epochs 50 --save-model --save-plot
```

### Option 3: Using Jupyter Notebooks

1. **Prepare data**: Run `Data_Preparation.ipynb`
2. **Train all models**: Run `All_Models_Training.ipynb`

## 📖 Usage

### Main Pipeline Script

The `main.py` script orchestrates the entire pipeline:

```bash
# Full pipeline: prepare data and train
python main.py --prepare-data --download --train --blocks 3 --variant batchnorm

# Only prepare data
python main.py --prepare-data --download

# Only train (data must be prepared first)
python main.py --train --blocks 3 --variant batchnorm --epochs 50

# Full pipeline with all options
python main.py --prepare-data --download --train --blocks 3 --variant batchnorm \
               --epochs 50 --batch-size 64 --save-model --save-plot
```

**Command Line Arguments:**

- **Pipeline Control**: `--prepare-data`, `--train`
- **Data Preparation Options**: `--download`, `--download-pickles`, `--img-size`, `--channels`, `--no-shuffle`, `--verify-only`
- **Training Options**: `--blocks`, `--variant`, `--epochs`, `--batch-size`, `--optimizer`, `--learning-rate`, `--test-size`, `--save-model`, `--save-plot`
- **Common Options**: `--base-dir`, `--output-dir`

### Data Preparation

The `data_preparation.py` script downloads and preprocesses the dataset.

#### Basic Usage

```bash
# Process existing dataset (if you already have CatsDogs/Cats/ and CatsDogs/Dogs/ folders)
python data_preparation.py

# Download and process dataset from Kaggle (requires Kaggle API setup)
python data_preparation.py --download

# Download pre-prepared pickle files from Google Drive (recommended alternative)
python data_preparation.py --download-pickles
```

#### Custom Options

```bash
# Specify custom base directory
python data_preparation.py --base-dir /path/to/dataset

# Use RGB images instead of grayscale
python data_preparation.py --channels 3

# Use larger image size (e.g., 150x150)
python data_preparation.py --img-size 150

# Don't shuffle the data
python data_preparation.py --no-shuffle

# Only verify existing pickle files
python data_preparation.py --verify-only
```

#### Complete Example

```bash
# Download dataset, process as RGB images with 150x150 size
python data_preparation.py --download --channels 3 --img-size 150
```

**Command Line Arguments:**

- `--download`: Download dataset from Kaggle (requires Kaggle API setup and accepting competition terms)
- `--download-pickles`: Download pre-prepared pickle files from Google Drive (recommended alternative, no API needed)
- `--base-dir PATH`: Base directory for dataset (default: auto-detect)
- `--img-size SIZE`: Image size in pixels (default: 100)
- `--channels CHANNELS`: Number of channels: 1 for grayscale, 3 for RGB (default: 1)
- `--no-shuffle`: Do not shuffle the data during preparation
- `--verify-only`: Only verify existing pickle files without reprocessing

**Output:**

The script creates:
- `CatsDogs/Pickles/X.pickle`: Processed image array (normalized to [0, 1])
- `CatsDogs/Pickles/y.pickle`: Labels array (0 for cats, 1 for dogs)

**Dataset Structure:**

The script expects/creates the following directory structure:

```
CatsDogs/
├── Cats/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   └── ...
├── Dogs/
│   ├── dog.0.jpg
│   ├── dog.1.jpg
│   └── ...
└── Pickles/
    ├── X.pickle
    └── y.pickle
```

### Model Training

The `train_test.py` script trains and evaluates CNN models.

#### Basic Usage

```bash
# Train a 3-block base model (default)
python train_test.py

# Train a specific model
python train_test.py --blocks 2 --variant dropout
python train_test.py --blocks 4 --variant batchnorm
```

#### Custom Training Parameters

```bash
# Train with custom epochs and batch size
python train_test.py --blocks 3 --variant batchnorm --epochs 100 --batch-size 32

# Use different optimizer
python train_test.py --blocks 3 --optimizer sgd --learning-rate 0.01

# Save model and plots
python train_test.py --blocks 3 --variant batchnorm --save-model --save-plot
```

**Command Line Arguments:**

**Model Selection:**
- `--blocks`: Number of convolutional blocks (1-6, default: 3)
- `--variant`: Model variant - `base`, `dropout`, or `batchnorm` (default: base)

**Training Parameters:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch-size`: Batch size for training (default: 64)
- `--optimizer`: Optimizer to use - `adam`, `sgd`, or `rmsprop` (default: adam)
- `--learning-rate`: Learning rate (default: optimizer default)
- `--test-size`: Proportion of data for testing (default: 0.2)

**Data and Output:**
- `--base-dir`: Base directory for dataset (default: auto-detect)
- `--output-dir`: Directory to save outputs (default: base_dir/Results)
- `--save-model`: Save the trained model as .h5 file
- `--save-plot`: Save training history plots

#### Examples

**Example 1: Quick test with 1-block model**
```bash
python train_test.py --blocks 1 --epochs 10 --batch-size 32
```

**Example 2: Full training with best practices**
```bash
python train_test.py \
    --blocks 3 \
    --variant batchnorm \
    --epochs 50 \
    --batch-size 64 \
    --optimizer adam \
    --save-model \
    --save-plot
```

**Example 3: Experiment with different optimizers**
```bash
# Adam optimizer
python train_test.py --blocks 3 --optimizer adam

# SGD optimizer
python train_test.py --blocks 3 --optimizer sgd --learning-rate 0.01

# RMSprop optimizer
python train_test.py --blocks 3 --optimizer rmsprop --learning-rate 1e-4
```

**Example 4: Compare model variants**
```bash
# Base model
python train_test.py --blocks 3 --variant base --save-model --save-plot

# With dropout
python train_test.py --blocks 3 --variant dropout --save-model --save-plot

# With batch normalization
python train_test.py --blocks 3 --variant batchnorm --save-model --save-plot
```

### Jupyter Notebooks

#### All Models Training Notebook

The `All_Models_Training.ipynb` notebook trains and evaluates all 18 models:

1. **Setup**: Loads data and configures training parameters
2. **Training**: Individual cells for each model (run all or select specific ones)
3. **Results**: Automatic comparison and visualization of all models

**Usage:**
- Run all cells to train all 18 models
- Run specific cells to train only selected models
- Results are automatically saved and compared

#### Data Preparation Notebook

The `Data_Preparation.ipynb` notebook provides an interactive way to prepare the dataset with download options and visualization.

#### Original Model Notebooks

The `notebooks/` folder contains the original individual model notebooks:
- `Model_cnn_one_block.ipynb` through `Model_cnn_six_block.ipynb`
- Each contains 3 model variants (base, dropout, batchnorm)

## 🧠 Models

The repository includes 18 different CNN architectures:

### Architecture Variants

Each architecture comes in 3 variants:

1. **Base**: Standard CNN without regularization
2. **Dropout**: Base architecture with dropout layers to prevent overfitting
3. **BatchNorm**: Base architecture with batch normalization and dropout

### Available Architectures

- **1-block**: Simple CNN (2 conv layers + 1 maxpool)
- **2-block**: 2 convolutional blocks
- **3-block**: 3 convolutional blocks (recommended starting point)
- **4-block**: 4 convolutional blocks
- **5-block**: 5 convolutional blocks
- **6-block**: 6 convolutional blocks (most complex)

### Model Selection

```bash
# Example: Train different variants
python train_test.py --blocks 3 --variant base
python train_test.py --blocks 3 --variant dropout
python train_test.py --blocks 3 --variant batchnorm
```

## 📊 Results

After training, results are saved in the `Results/` directory:

```
Results/
├── model_3block_batchnorm.h5          # Saved model
├── model_3block_batchnorm_history.png # Training plots
└── model_3block_batchnorm_metrics.txt # Evaluation metrics
```

When training all models via the notebook, you'll also get:
- `all_models_results.csv`: Comparison table of all models
- `all_models_comparison.png`: Visual comparison chart
- Individual history plots for each model

## 🔧 Configuration

### Environment Detection

The scripts automatically detect the environment:
- **Local**: Uses `CatsDogs/` in current directory
- **Google Colab**: Uses `/content/drive/MyDrive/CatsDogs`

### Custom Paths

You can specify custom paths:

```bash
python main.py --base-dir /path/to/dataset --prepare-data --train
```

### Alternative Data Sources

If you prefer not to download from Kaggle, you can use pre-prepared data:

**Option 1: Automated Download (Recommended)**
```bash
python data_preparation.py --download-pickles
```
This automatically downloads the pre-prepared pickle files from Google Drive.

**Option 2: Manual Download**

**Original Project Resources:**
- **Notebooks repository**: https://drive.google.com/drive/folders/1m3UukrQ4htoX6rj_44pp2as6Af8N2Cvq?usp=share_link
- **CatsDogs repository** (with all files): https://drive.google.com/drive/folders/1wDN-xlK4YvsqEa0t2lLZ0RSpFJU55ahW?usp=share_link
- **X.pickle** (images): https://drive.google.com/file/d/1-ZQh_Ch7SwgSfBCw6TXOLgO8ZKgVxdS2/view?usp=share_link
- **y.pickle** (labels): https://drive.google.com/file/d/1-b38kY0LIErEWYR6VvNmxBePEDfQeZhd/view?usp=share_link

Simply download and place the pickle files in `CatsDogs/Pickles/` directory.

## 🐛 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**
   - **Solution**: Install dependencies with `pip install -r requirements.txt`

2. **Kaggle API Error / Missing Credentials / 403 Forbidden**
   - **Solution**: Follow the detailed setup instructions in the Installation section above
   - Quick fix: Ensure `kaggle.json` is in `~/.kaggle/kaggle.json` (Mac/Linux) or `C:\Users\<username>\.kaggle\kaggle.json` (Windows)
   - **File not found**: Make sure the file is named exactly `kaggle.json` (lowercase)
   - **Permission denied**: On Mac/Linux, set permissions with `chmod 600 ~/.kaggle/kaggle.json`
   - **403 Forbidden**: You need to accept the competition terms at https://www.kaggle.com/c/dogs-vs-cats
   - **Alternative (Recommended)**: Use `python data_preparation.py --download-pickles` to download pre-prepared pickles from Google Drive (no API needed)

3. **Data Not Found**
   - **Solution**: Run data preparation first:
     - `python data_preparation.py --download` (requires Kaggle API)
     - Or `python data_preparation.py --download-pickles` (recommended, no API needed)
   - Or manually download pre-prepared pickles from the Google Drive links above

4. **Memory Errors**
   - **Solution**: Reduce batch size or use a smaller model
   - Try: `--batch-size 32` or `--blocks 1`

5. **CUDA/GPU Issues**
   - **Solution**: TensorFlow will automatically use CPU if GPU is unavailable
   - For GPU support, ensure CUDA and cuDNN are properly installed

6. **Directory Not Found**
   - **Solution**: Use `--base-dir` to specify the correct path
   - Or ensure you're running from the project root directory

7. **Slow Training**
   - **Solution**: Use GPU if available, or reduce model complexity
   - Consider using fewer epochs for initial testing

8. **Poor Accuracy**
   - **Solution**: Try different variants (dropout/batchnorm) or more epochs
   - Experiment with different optimizers and learning rates

## 📝 Dataset

The project uses the [Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats) dataset from Kaggle:

- **25,000 images** (12,500 cats, 12,500 dogs)
- **Format**: JPEG images
- **Size**: Variable (resized to 100x100 by default)
- **Channels**: Grayscale (1 channel) by default, RGB (3 channels) optional

### Dataset License

Please refer to the original Kaggle competition page for dataset license terms.

## 💡 Tips

1. **Start simple**: Begin with a 1-2 block model to test your setup
2. **Use batchnorm**: Models with batch normalization often perform better
3. **Monitor overfitting**: Watch the gap between training and validation accuracy
4. **Save models**: Use `--save-model` to keep trained models for later use
5. **Experiment**: Try different optimizers and learning rates to find what works best
6. **Use notebooks**: The Jupyter notebooks provide interactive exploration and visualization

## 👤 Author

**Lorenzo Rossi** - Student 982595

## 🙏 Acknowledgments

- Kaggle for providing the Dogs vs Cats dataset
- TensorFlow/Keras for the deep learning framework
- University course materials and guidance

## 📄 License

This project is part of an academic research project. Please refer to the original dataset license for usage terms.

---

**Note**: This is an optimized Python version of the original Jupyter notebooks. The notebooks are still available in the `notebooks/` folder for reference and interactive exploration.
