#!/usr/bin/env python3
"""
Main Pipeline Script for Cats vs Dogs Classification

This script orchestrates the complete pipeline:
1. Data preparation (download and preprocessing)
2. Model training and evaluation

It can run the full pipeline or individual steps based on command-line arguments.

Usage:
    # Run full pipeline
    python main.py --prepare-data --train --blocks 3 --variant batchnorm
    
    # Only prepare data
    python main.py --prepare-data --download
    
    # Only train (assumes data is already prepared)
    python main.py --train --blocks 3 --variant batchnorm
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_preparation(args):
    """Run data preparation script."""
    logger.info("=" * 60)
    logger.info("STEP 1: Data Preparation")
    logger.info("=" * 60)
    
    cmd = [sys.executable, 'data_preparation.py']
    
    if args.download:
        cmd.append('--download')
    if args.download_pickles:
        cmd.append('--download-pickles')
    if args.base_dir:
        cmd.extend(['--base-dir', args.base_dir])
    if args.img_size:
        cmd.extend(['--img-size', str(args.img_size)])
    if args.channels:
        cmd.extend(['--channels', str(args.channels)])
    if args.no_shuffle:
        cmd.append('--no-shuffle')
    if args.verify_only:
        cmd.append('--verify-only')
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        logger.error("Data preparation failed!")
        return False
    
    logger.info("Data preparation completed successfully!")
    return True


def run_training(args):
    """Run training script."""
    logger.info("=" * 60)
    logger.info("STEP 2: Model Training and Evaluation")
    logger.info("=" * 60)
    
    cmd = [sys.executable, 'train_test.py']
    
    if args.blocks:
        cmd.extend(['--blocks', str(args.blocks)])
    if args.variant:
        cmd.extend(['--variant', args.variant])
    if args.base_dir:
        cmd.extend(['--base-dir', args.base_dir])
    if args.epochs:
        cmd.extend(['--epochs', str(args.epochs)])
    if args.batch_size:
        cmd.extend(['--batch-size', str(args.batch_size)])
    if args.optimizer:
        cmd.extend(['--optimizer', args.optimizer])
    if args.learning_rate:
        cmd.extend(['--learning-rate', str(args.learning_rate)])
    if args.test_size:
        cmd.extend(['--test-size', str(args.test_size)])
    if args.save_model:
        cmd.append('--save-model')
    if args.save_plot:
        cmd.append('--save-plot')
    if args.output_dir:
        cmd.extend(['--output-dir', args.output_dir])
    
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode != 0:
        logger.error("Training failed!")
        return False
    
    logger.info("Training completed successfully!")
    return True


def main():
    """Main function to orchestrate the pipeline."""
    parser = argparse.ArgumentParser(
        description='Cats vs Dogs Classification Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline: prepare data and train
  python main.py --prepare-data --download --train --blocks 3 --variant batchnorm
  
  # Prepare data using Google Drive pickles (no Kaggle API needed)
  python main.py --prepare-data --download-pickles
  
  # Only prepare data from Kaggle
  python main.py --prepare-data --download
  
  # Only train (data must be prepared first)
  python main.py --train --blocks 3 --variant batchnorm --epochs 50
  
  # Full pipeline with all options
  python main.py --prepare-data --download --train --blocks 3 --variant batchnorm \\
                 --epochs 50 --batch-size 64 --save-model --save-plot
        """
    )
    
    # Pipeline control
    parser.add_argument(
        '--prepare-data',
        action='store_true',
        help='Run data preparation step'
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training step'
    )
    
    # Data preparation arguments
    data_group = parser.add_argument_group('Data Preparation Options')
    data_group.add_argument(
        '--download',
        action='store_true',
        help='Download dataset from Kaggle (requires Kaggle API)'
    )
    data_group.add_argument(
        '--download-pickles',
        action='store_true',
        help='Download pre-prepared pickle files from Google Drive (recommended alternative, no API needed)'
    )
    data_group.add_argument(
        '--img-size',
        type=int,
        help='Image size in pixels (default: 100)'
    )
    data_group.add_argument(
        '--channels',
        type=int,
        choices=[1, 3],
        help='Number of channels: 1 for grayscale, 3 for RGB (default: 1)'
    )
    data_group.add_argument(
        '--no-shuffle',
        action='store_true',
        help='Do not shuffle the data during preparation'
    )
    data_group.add_argument(
        '--verify-only',
        action='store_true',
        help='Only verify existing pickle files'
    )
    
    # Training arguments
    train_group = parser.add_argument_group('Training Options')
    train_group.add_argument(
        '--blocks',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Number of convolutional blocks (default: 3)'
    )
    train_group.add_argument(
        '--variant',
        type=str,
        choices=['base', 'dropout', 'batchnorm'],
        help='Model variant: base, dropout, or batchnorm (default: base)'
    )
    train_group.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (default: 50)'
    )
    train_group.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for training (default: 64)'
    )
    train_group.add_argument(
        '--optimizer',
        type=str,
        choices=['adam', 'sgd', 'rmsprop'],
        help='Optimizer to use (default: adam)'
    )
    train_group.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (default: optimizer default)'
    )
    train_group.add_argument(
        '--test-size',
        type=float,
        help='Proportion of data for testing (default: 0.2)'
    )
    train_group.add_argument(
        '--save-model',
        action='store_true',
        help='Save the trained model'
    )
    train_group.add_argument(
        '--save-plot',
        action='store_true',
        help='Save training history plots'
    )
    
    # Common arguments
    common_group = parser.add_argument_group('Common Options')
    common_group.add_argument(
        '--base-dir',
        type=str,
        help='Base directory for dataset (default: auto-detect)'
    )
    common_group.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save outputs (default: base_dir/Results)'
    )
    
    args = parser.parse_args()
    
    # Check if at least one step is requested
    if not args.prepare_data and not args.train:
        parser.print_help()
        logger.error("\nError: You must specify at least one step: --prepare-data or --train")
        sys.exit(1)
    
    # Run pipeline steps
    success = True
    
    if args.prepare_data:
        success = run_data_preparation(args)
        if not success:
            logger.error("Pipeline failed at data preparation step.")
            sys.exit(1)
    
    if args.train:
        # Check if data exists (unless we just prepared it)
        if not args.prepare_data:
            base_dir = args.base_dir if args.base_dir else os.path.join(os.getcwd(), 'CatsDogs')
            pickle_dir = os.path.join(base_dir, 'Pickles')
            if not os.path.exists(os.path.join(pickle_dir, 'X.pickle')):
                logger.error(
                    f"Data not found at {pickle_dir}. "
                    "Please run data preparation first with: python main.py --prepare-data"
                )
                sys.exit(1)
        
        success = run_training(args)
        if not success:
            logger.error("Pipeline failed at training step.")
            sys.exit(1)
    
    if success:
        logger.info("=" * 60)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 60)


if __name__ == '__main__':
    main()

