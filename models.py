"""
CNN Models for Cats vs Dogs Classification

This module contains all CNN model architectures used for binary image classification.
Each model comes in three variants:
- Base: Standard architecture
- Dropout: Base architecture with dropout layers
- BatchNorm: Base architecture with batch normalization and dropout

Models are organized by the number of convolutional blocks (1-6 blocks).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple


def create_one_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a one-block CNN model (base architecture).
    
    Architecture:
    - Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Flatten -> Dense(128) -> Dense(1, sigmoid)
    
    Args:
        input_shape: Shape of input images (height, width, channels)
        
    Returns:
        Compiled Keras model
    """
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_one_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a one-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_one_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a one-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_two_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a two-block CNN model (base architecture).
    
    Architecture:
    - Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
    - Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Flatten -> Dense(128) -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_two_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a two-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_two_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a two-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_three_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a three-block CNN model (base architecture).
    
    Architecture:
    - Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
    - Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Flatten -> Dense(256) -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(256, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_three_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a three-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_three_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a three-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_four_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a four-block CNN model (base architecture).
    
    Architecture:
    - Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
    - Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Block 4: Conv2D(256) -> Conv2D(256) -> MaxPool2D
    - Flatten -> Dense(512) -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_four_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a four-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_four_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a four-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_five_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a five-block CNN model (base architecture).
    
    Architecture:
    - Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
    - Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Block 4: Conv2D(256) -> Conv2D(256) -> MaxPool2D
    - Block 5: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Flatten -> Dense(512) -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_five_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a five-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_five_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a five-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_six_block_base(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Create a six-block CNN model (base architecture).
    
    Architecture:
    - Block 1: Conv2D(32) -> Conv2D(32) -> MaxPool2D
    - Block 2: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Block 3: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Block 4: Conv2D(256) -> Conv2D(256) -> MaxPool2D
    - Block 5: Conv2D(128) -> Conv2D(128) -> MaxPool2D
    - Block 6: Conv2D(64) -> Conv2D(64) -> MaxPool2D
    - Flatten -> Dense(512) -> Dense(1, sigmoid)
    """
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_six_block_dropout(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a six-block CNN model with dropout layers."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


def create_six_block_batchnorm(input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """Create a six-block CNN model with batch normalization and dropout."""
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, 
                     input_shape=input_shape, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(512, use_bias=False, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


# Model factory function
def get_model(blocks: int, variant: str = 'base', input_shape: Tuple[int, int, int] = (100, 100, 1)) -> keras.Model:
    """
    Factory function to get a model by number of blocks and variant.
    
    Args:
        blocks: Number of convolutional blocks (1-6)
        variant: Model variant ('base', 'dropout', 'batchnorm')
        input_shape: Shape of input images
        
    Returns:
        Keras model
        
    Raises:
        ValueError: If blocks or variant is invalid
    """
    variant = variant.lower()
    
    if blocks == 1:
        if variant == 'base':
            return create_one_block_base(input_shape)
        elif variant == 'dropout':
            return create_one_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_one_block_batchnorm(input_shape)
    elif blocks == 2:
        if variant == 'base':
            return create_two_block_base(input_shape)
        elif variant == 'dropout':
            return create_two_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_two_block_batchnorm(input_shape)
    elif blocks == 3:
        if variant == 'base':
            return create_three_block_base(input_shape)
        elif variant == 'dropout':
            return create_three_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_three_block_batchnorm(input_shape)
    elif blocks == 4:
        if variant == 'base':
            return create_four_block_base(input_shape)
        elif variant == 'dropout':
            return create_four_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_four_block_batchnorm(input_shape)
    elif blocks == 5:
        if variant == 'base':
            return create_five_block_base(input_shape)
        elif variant == 'dropout':
            return create_five_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_five_block_batchnorm(input_shape)
    elif blocks == 6:
        if variant == 'base':
            return create_six_block_base(input_shape)
        elif variant == 'dropout':
            return create_six_block_dropout(input_shape)
        elif variant == 'batchnorm':
            return create_six_block_batchnorm(input_shape)
    else:
        raise ValueError(f"Invalid number of blocks: {blocks}. Must be between 1 and 6.")
    
    raise ValueError(f"Invalid variant: {variant}. Must be 'base', 'dropout', or 'batchnorm'.")

