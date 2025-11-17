#!/usr/bin/env python3
"""
Training and Testing Script for Cats vs Dogs CNN Models

This script loads preprocessed data, trains CNN models, and evaluates their performance.
It supports training any model variant from the models.py module.

Usage:
    python train_test.py --blocks 3 --variant batchnorm --epochs 50 --batch-size 64
"""

import os
import sys
import argparse
import pickle
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split

from models import get_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set TensorFlow to use float64 (matching original notebooks)
tf.keras.backend.set_floatx("float64")


def detect_environment() -> str:
    """Detect if running in Google Colab or local environment."""
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
    
    return base_dir


def load_data(base_dir: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load preprocessed data from pickle files.
    
    Args:
        base_dir: Base directory containing Pickles folder
        
    Returns:
        Tuple of (X, y) arrays
    """
    pickles_dir = os.path.join(base_dir, 'Pickles')
    x_pickle_path = os.path.join(pickles_dir, 'X.pickle')
    y_pickle_path = os.path.join(pickles_dir, 'y.pickle')
    
    if not os.path.exists(x_pickle_path) or not os.path.exists(y_pickle_path):
        raise FileNotFoundError(
            f"Pickle files not found. Please run data_preparation.py first.\n"
            f"Expected files: {x_pickle_path}, {y_pickle_path}"
        )
    
    logger.info("Loading preprocessed data...")
    
    with open(x_pickle_path, 'rb') as f:
        X = pickle.load(f)
    
    with open(y_pickle_path, 'rb') as f:
        y = pickle.load(f)
    
    logger.info(f"Loaded data: X shape = {X.shape}, y shape = {y.shape}")
    
    return X, y


def prepare_data(X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
    """
    Prepare data for training: normalize and split into train/test sets.
    
    Args:
        X: Image array
        y: Label array
        test_size: Proportion of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    logger.info("Preparing data for training...")
    
    # Convert to TensorFlow tensors and normalize (if not already normalized)
    # Note: data_preparation.py already normalizes to [0, 1], but we ensure it here
    X = tf.cast(X, tf.float32)
    if X.numpy().max() > 1.0:
        X = X / 255.0
        logger.info("Normalized pixel values to [0, 1]")
    
    # One-hot encode labels (depth=1 for binary classification)
    x_size = X.shape[0]
    depth = 1
    y = tf.reshape(tf.one_hot(y, depth), shape=[x_size, depth])
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X.numpy(), y.numpy(), test_size=test_size, random_state=42
    )
    
    logger.info(f"Train set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test


def performance_plot(history: keras.callbacks.History, save_path: Optional[str] = None):
    """
    Plot training history (loss and accuracy).
    
    Args:
        history: Keras training history object
        save_path: Optional path to save the plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='train')
    axes[0].plot(history.history['val_loss'], label='val')
    axes[0].set_xlabel('Epoch', size=12)
    axes[0].set_ylabel('Loss', size=12)
    axes[0].legend(fontsize=12)
    axes[0].set_title('Model Loss')
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history.history['binary_accuracy'], label='train')
    axes[1].plot(history.history['val_binary_accuracy'], label='val')
    axes[1].set_xlabel('Epoch', size=12)
    axes[1].set_ylabel('Accuracy', size=12)
    axes[1].legend(fontsize=12)
    axes[1].set_title('Model Accuracy')
    axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 50,
    batch_size: int = 64,
    optimizer: str = 'adam',
    learning_rate: float = None
) -> keras.callbacks.History:
    """
    Train a model.
    
    Args:
        model: Keras model to train
        X_train: Training images
        y_train: Training labels
        X_test: Test images
        y_test: Test labels
        epochs: Number of training epochs
        batch_size: Batch size for training
        optimizer: Optimizer name ('adam', 'sgd', 'rmsprop')
        learning_rate: Learning rate (if None, uses default)
        
    Returns:
        Training history
    """
    # Setup optimizer
    if optimizer.lower() == 'adam':
        opt = keras.optimizers.Adam(learning_rate=learning_rate) if learning_rate else keras.optimizers.Adam()
    elif optimizer.lower() == 'sgd':
        lr = learning_rate if learning_rate else 0.01
        opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, decay=lr/epochs)
    elif optimizer.lower() == 'rmsprop':
        lr = learning_rate if learning_rate else 1e-3
        opt = keras.optimizers.RMSprop(learning_rate=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    
    # Compile model
    loss = keras.losses.BinaryCrossentropy(from_logits=False)  # from_logits=False because we use sigmoid
    model.compile(optimizer=opt, loss=loss, metrics=['binary_accuracy'])
    
    logger.info(f"Training model: {model.name}")
    logger.info(f"Optimizer: {optimizer}, Epochs: {epochs}, Batch size: {batch_size}")
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        validation_data=(X_test, y_test)
    )
    
    return history


def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> dict:
    """
    Evaluate a trained model.
    
    Args:
        model: Trained Keras model
        X_test: Test images
        y_test: Test labels
        
    Returns:
        Dictionary with evaluation metrics
    """
    logger.info("Evaluating model on test set...")
    
    results = model.evaluate(X_test, y_test, verbose=0)
    
    metrics = {
        'test_loss': results[0],
        'test_accuracy': results[1]
    }
    
    logger.info(f"Test Loss: {metrics['test_loss']:.4f}")
    logger.info(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
    
    return metrics


def save_model(model: keras.Model, save_path: str):
    """
    Save a trained model.
    
    Args:
        model: Trained Keras model
        save_path: Path to save the model
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    logger.info(f"Model saved to {save_path}")


def main():
    """Main function to run training and testing."""
    parser = argparse.ArgumentParser(
        description='Train and test CNN models for Cats vs Dogs classification'
    )
    parser.add_argument(
        '--blocks',
        type=int,
        default=3,
        choices=[1, 2, 3, 4, 5, 6],
        help='Number of convolutional blocks (default: 3)'
    )
    parser.add_argument(
        '--variant',
        type=str,
        default='base',
        choices=['base', 'dropout', 'batchnorm'],
        help='Model variant (default: base)'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default=None,
        help='Base directory for dataset (default: auto-detect)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs (default: 50)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size for training (default: 64)'
    )
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        choices=['adam', 'sgd', 'rmsprop'],
        help='Optimizer to use (default: adam)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=None,
        help='Learning rate (default: optimizer default)'
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help='Proportion of data for testing (default: 0.2)'
    )
    parser.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save training history plots'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save outputs (default: base_dir/Results)'
    )
    
    args = parser.parse_args()
    
    # Detect environment and set base directory
    default_base_dir = detect_environment()
    base_dir = args.base_dir if args.base_dir else default_base_dir
    
    logger.info(f"Base directory: {base_dir}")
    logger.info(f"Model: {args.blocks}-block, variant: {args.variant}")
    
    # Load data
    try:
        X, y = load_data(base_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(X, y, test_size=args.test_size)
    
    # Get model
    input_shape = X_train.shape[1:]
    logger.info(f"Input shape: {input_shape}")
    
    try:
        model = get_model(args.blocks, args.variant, input_shape)
        logger.info(f"Model created: {model.name}")
        model.summary()
    except ValueError as e:
        logger.error(f"Error creating model: {e}")
        sys.exit(1)
    
    # Train model
    try:
        history = train_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            epochs=args.epochs,
            batch_size=args.batch_size,
            optimizer=args.optimizer,
            learning_rate=args.learning_rate
        )
    except Exception as e:
        logger.error(f"Error during training: {e}")
        sys.exit(1)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save outputs
    output_dir = args.output_dir if args.output_dir else os.path.join(base_dir, 'Results')
    os.makedirs(output_dir, exist_ok=True)
    
    model_name = f"model_{args.blocks}block_{args.variant}"
    
    if args.save_plot:
        plot_path = os.path.join(output_dir, f"{model_name}_history.png")
        performance_plot(history, save_path=plot_path)
    else:
        performance_plot(history)
    
    if args.save_model:
        model_path = os.path.join(output_dir, f"{model_name}.h5")
        save_model(model, model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Model: {args.blocks}-block {args.variant}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Optimizer: {args.optimizer}\n")
        f.write(f"\nTest Loss: {metrics['test_loss']:.4f}\n")
        f.write(f"Test Accuracy: {metrics['test_accuracy']:.4f}\n")
        f.write(f"\nFinal Training Loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"Final Training Accuracy: {history.history['binary_accuracy'][-1]:.4f}\n")
        f.write(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Validation Accuracy: {history.history['val_binary_accuracy'][-1]:.4f}\n")
    
    logger.info(f"Metrics saved to {metrics_path}")
    logger.info("Training and evaluation completed successfully!")


if __name__ == '__main__':
    main()

