#!/usr/bin/env python3
"""
Data Preparation Script for Cats vs Dogs Dataset

This script downloads (optional), processes, and prepares the Cats vs Dogs dataset
for machine learning tasks. It supports both Google Colab and local environments.

Usage:
    python data_preparation.py [--download] [--base-dir BASE_DIR] [--img-size SIZE] [--channels CHANNELS]

Author: Optimized version of Data_Preparation.ipynb
"""

import os
import sys
import argparse
import random
import pickle
import zipfile
import shutil
import glob
from pathlib import Path
from typing import Tuple, List, Optional
import logging

import numpy as np
import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def detect_environment() -> Tuple[bool, str]:
    """
    Detect if running in Google Colab or local environment.
    
    Returns:
        Tuple of (is_colab, base_directory)
    """
    try:
        import google.colab
        is_colab = True
        try:
            from google.colab import drive
            drive.mount('/content/drive', force_remount=False)
            base_dir = '/content/drive/MyDrive/CatsDogs'
        except Exception as e:
            logger.warning(f"Could not mount Google Drive: {e}")
            base_dir = '/content/CatsDogs'
    except ImportError:
        is_colab = False
        base_dir = os.path.join(os.getcwd(), 'CatsDogs')
    
    return is_colab, base_dir


def setup_kaggle_api() -> bool:
    """
    Setup Kaggle API credentials.
    
    Returns:
        True if setup successful, False otherwise
    """
    kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
    kaggle_json = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_json):
        logger.error(f"Kaggle API credentials not found at {kaggle_json}")
        logger.info("Please download your kaggle.json from https://www.kaggle.com/account")
        logger.info(f"and place it at: {kaggle_json}")
        return False
    
    os.makedirs(kaggle_dir, exist_ok=True)
    # Set proper permissions on Unix systems
    if sys.platform != 'win32':
        os.chmod(kaggle_json, 0o600)
    
    return True


def download_dataset(download_dir: str, is_colab: bool = False) -> bool:
    """
    Download the Dogs vs Cats dataset from Kaggle.
    
    Args:
        download_dir: Directory to download the dataset to
        is_colab: Whether running in Google Colab
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        import subprocess
        
        logger.info("Downloading dataset from Kaggle...")
        logger.info(f"Download directory: {download_dir}")
        
        # Use subprocess for both Colab and local
        result = subprocess.run(
            ['kaggle', 'competitions', 'download', '-c', 'dogs-vs-cats'],
            cwd=download_dir,
            check=True,
            capture_output=True,
            text=True
        )
        
        logger.info("Dataset downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to download dataset: {e}")
        if e.stderr:
            logger.error(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        logger.error("Kaggle CLI not found. Please install it with: pip install kaggle")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during download: {e}")
        return False


def extract_and_organize_dataset(download_dir: str, base_dir: str) -> bool:
    """
    Extract and organize the dataset into Cats and Dogs folders.
    
    Args:
        download_dir: Directory containing the downloaded zip files
        base_dir: Base directory to organize the dataset
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create directory structure
        cats_dir = os.path.join(base_dir, 'Cats')
        dogs_dir = os.path.join(base_dir, 'Dogs')
        pickles_dir = os.path.join(base_dir, 'Pickles')
        
        os.makedirs(cats_dir, exist_ok=True)
        os.makedirs(dogs_dir, exist_ok=True)
        os.makedirs(pickles_dir, exist_ok=True)
        
        # Extract dogs-vs-cats.zip
        zip_path = os.path.join(download_dir, 'dogs-vs-cats.zip')
        if os.path.exists(zip_path):
            logger.info("Extracting dogs-vs-cats.zip...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
        
        # Extract train.zip
        train_zip_path = os.path.join(download_dir, 'train.zip')
        train_dir = os.path.join(download_dir, 'train')
        
        if os.path.exists(train_zip_path):
            logger.info("Extracting train.zip...")
            with zipfile.ZipFile(train_zip_path, 'r') as zip_ref:
                zip_ref.extractall(download_dir)
        
        if not os.path.exists(train_dir):
            logger.error(f"Training directory not found at {train_dir}")
            return False
        
        # Organize images
        logger.info("Organizing images into Cats and Dogs folders...")
        
        cat_files = glob.glob(os.path.join(train_dir, 'cat.*'))
        dog_files = glob.glob(os.path.join(train_dir, 'dog.*'))
        
        # Move cat images
        for file in cat_files:
            shutil.move(file, cats_dir)
        
        # Move dog images
        for file in dog_files:
            shutil.move(file, dogs_dir)
        
        logger.info(f"Organized {len(cat_files)} cat images and {len(dog_files)} dog images")
        return True
        
    except Exception as e:
        logger.error(f"Error extracting and organizing dataset: {e}")
        return False


def load_and_process_images(
    base_dir: str,
    img_size: Tuple[int, int] = (100, 100),
    channels: int = 1,
    shuffle: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load and process images from the dataset.
    
    Args:
        base_dir: Base directory containing Cats and Dogs folders
        img_size: Target image size (width, height)
        channels: Number of channels (1 for grayscale, 3 for RGB)
        shuffle: Whether to shuffle the data
        
    Returns:
        Tuple of (X, y) where X is the image array and y is the labels
    """
    img_width, img_height = img_size
    categories = ['Cats', 'Dogs']
    pets = []
    
    # Determine OpenCV read mode
    read_mode = cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR
    
    logger.info(f"Loading images from {base_dir}...")
    logger.info(f"Image size: {img_size}, Channels: {channels}")
    
    for category in categories:
        path = os.path.join(base_dir, category)
        
        if not os.path.exists(path):
            logger.warning(f"Directory {path} does not exist. Skipping...")
            continue
        
        pet_class = categories.index(category)
        image_files = [f for f in os.listdir(path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        logger.info(f"Processing {len(image_files)} {category.lower()} images...")
        
        for img_file in image_files:
            img_path = os.path.join(path, img_file)
            try:
                img_array = cv2.imread(img_path, read_mode)
                
                if img_array is None:
                    logger.debug(f"Could not load image: {img_path}")
                    continue
                
                # Resize image
                resized = cv2.resize(img_array, img_size)
                pets.append([resized, pet_class])
                
            except Exception as e:
                logger.debug(f"Error processing {img_path}: {e}")
                continue
    
    logger.info(f"Total images loaded: {len(pets)}")
    
    if shuffle:
        logger.info("Shuffling data...")
        random.shuffle(pets)
    
    # Split into X and y
    X = []
    y = []
    
    for image, label in pets:
        X.append(image)
        y.append(label)
    
    # Convert to numpy arrays
    X = np.array(X, dtype=np.float64)
    X = X.reshape(-1, img_height, img_width, channels)
    y = np.array(y, dtype=np.int32)
    
    # Normalize pixel values to [0, 1]
    X = X / 255.0
    
    logger.info(f"Data shape: X = {X.shape}, y = {y.shape}")
    logger.info(f"X dtype: {X.dtype}, y dtype: {y.dtype}")
    logger.info(f"X range: [{X.min():.3f}, {X.max():.3f}]")
    
    return X, y


def save_pickles(X: np.ndarray, y: np.ndarray, base_dir: str) -> bool:
    """
    Save processed data as pickle files.
    
    Args:
        X: Image array
        y: Label array
        base_dir: Base directory to save pickles
        
    Returns:
        True if successful, False otherwise
    """
    try:
        pickles_dir = os.path.join(base_dir, 'Pickles')
        os.makedirs(pickles_dir, exist_ok=True)
        
        x_pickle_path = os.path.join(pickles_dir, 'X.pickle')
        y_pickle_path = os.path.join(pickles_dir, 'y.pickle')
        
        logger.info(f"Saving X to {x_pickle_path}...")
        with open(x_pickle_path, 'wb') as f:
            pickle.dump(X, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saving y to {y_pickle_path}...")
        with open(y_pickle_path, 'wb') as f:
            pickle.dump(y, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Verify saved files
        file_size_x = os.path.getsize(x_pickle_path) / (1024 * 1024)  # MB
        file_size_y = os.path.getsize(y_pickle_path) / (1024 * 1024)  # MB
        
        logger.info(f"Pickle files saved successfully!")
        logger.info(f"X.pickle size: {file_size_x:.2f} MB")
        logger.info(f"y.pickle size: {file_size_y:.2f} MB")
        
        return True
        
    except Exception as e:
        logger.error(f"Error saving pickles: {e}")
        return False


def load_pickles(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load pickled data.
    
    Args:
        base_dir: Base directory containing Pickles folder
        
    Returns:
        Tuple of (X, y)
    """
    pickles_dir = os.path.join(base_dir, 'Pickles')
    x_pickle_path = os.path.join(pickles_dir, 'X.pickle')
    y_pickle_path = os.path.join(pickles_dir, 'y.pickle')
    
    logger.info("Loading pickled data...")
    
    with open(x_pickle_path, 'rb') as f:
        X = pickle.load(f)
    
    with open(y_pickle_path, 'rb') as f:
        y = pickle.load(f)
    
    logger.info(f"Loaded data: X = {X.shape}, y = {y.shape}")
    
    return X, y


def main():
    """Main function to run the data preparation pipeline."""
    parser = argparse.ArgumentParser(
        description='Prepare Cats vs Dogs dataset for machine learning'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='Download dataset from Kaggle (requires Kaggle API setup)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory for dataset (default: auto-detect)'
    )
    parser.add_argument(
        '--img-size',
        type=int,
        default=100,
        help='Image size (width and height, default: 100)'
    )
    parser.add_argument(
        '--channels',
        type=int,
        default=1,
        choices=[1, 3],
        help='Number of channels: 1 for grayscale, 3 for RGB (default: 1)'
    )
    parser.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the data'
    )
    parser.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing pickle files without reprocessing'
    )
    
    args = parser.parse_args()
    
    # Detect environment
    is_colab, default_base_dir = detect_environment()
    base_dir = args.base_dir if args.base_dir else default_base_dir
    
    logger.info(f"Environment: {'Google Colab' if is_colab else 'Local'}")
    logger.info(f"Base directory: {base_dir}")
    
    # Download dataset if requested
    if args.download:
        if not setup_kaggle_api():
            logger.error("Kaggle API setup failed. Cannot download dataset.")
            sys.exit(1)
        
        download_dir = '/content' if is_colab else os.getcwd()
        
        if not download_dataset(download_dir, is_colab):
            logger.error("Dataset download failed.")
            sys.exit(1)
        
        if not extract_and_organize_dataset(download_dir, base_dir):
            logger.error("Dataset extraction and organization failed.")
            sys.exit(1)
    
    # Verify only mode
    if args.verify_only:
        try:
            X, y = load_pickles(base_dir)
            logger.info("Verification successful!")
            return
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            sys.exit(1)
    
    # Process images
    img_size = (args.img_size, args.img_size)
    
    try:
        X, y = load_and_process_images(
            base_dir=base_dir,
            img_size=img_size,
            channels=args.channels,
            shuffle=not args.no_shuffle
        )
    except Exception as e:
        logger.error(f"Error processing images: {e}")
        sys.exit(1)
    
    # Save pickles
    if not save_pickles(X, y, base_dir):
        logger.error("Failed to save pickle files.")
        sys.exit(1)
    
    # Verify saved files
    try:
        X_loaded, y_loaded = load_pickles(base_dir)
        logger.info("Data preparation completed successfully!")
        logger.info(f"Final verification: X = {X_loaded.shape}, y = {y_loaded.shape}")
    except Exception as e:
        logger.warning(f"Could not verify saved files: {e}")


if __name__ == '__main__':
    main()

