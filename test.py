import os
import yaml
import pandas as pd
import glob
from pathlib import Path
import logging
from typing import List, Tuple
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    Trainer, 
    TrainingArguments
)

from .src.Utils import get_dataset_files, load_and_preprocess_dataset, get_training_args
from .src.dataset import create_datasets
from .src.metrics import compute_metrics, plot_confusion_matrix


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# testing individual datasets
def test_individual(config: dict):
    """Test the model one by one on multiple datasets."""
    try:
        dataset_files = get_dataset_files(config['data']['data_dir'], logger)
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(config['testing']['trained_model_path'])
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # traverse through each datasets
        for file_path in dataset_files:
            
            dataset_name = Path(file_path).stem
            logger.info(f"\nStarting training on dataset: {dataset_name}")
            
            # Create dataset-specific output directories
            test_results_dir = os.path.join(config['testing']['individual_save_dir'], dataset_name)
            
            # Load and preprocess current dataset
            df = load_and_preprocess_dataset(file_path, logger)
            _, test_dataset = create_datasets(df, config, logger, 'testing')
            
            training_args = get_training_args(config, test_results_dir)
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=_,
                eval_dataset=test_dataset,
                compute_metrics=compute_metrics
            )
            
            # Save model and evaluation results
            trainer.evaluate()
            
            # Save evaluation results
            plot_confusion_matrix(
                test_dataset, 
                trainer, 
                os.path.join(test_results_dir, 'confusion_matrix.png')
            )
            
    except Exception as e:
        logger.error(f"Individual testing failed: {str(e)}")
        raise


# trainning on all datasets combined
def test_combined(config: dict):
    """Test on test set from all datasets combined."""
    try:
        dataset_files = get_dataset_files(config['data']['data_dir'], logger)
        
        # Combine all datasets
        dfs = []
        for file_path in dataset_files:
            df = load_and_preprocess_dataset(file_path, logger)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        
        # Create combined dataset
        _, test_dataset = create_datasets(combined_df, config, logger, 'testing')
        
        # Load the model
        model = AutoModelForSequenceClassification.from_pretrained(config['testing']['trained_model_path'])
        
        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
            
        training_args = get_training_args(config, None)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=_,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics
        )
        
        # Evaluate the model
        trainer.evaluate()
        
        # Save evaluation results
        plot_confusion_matrix(
            test_dataset, 
            trainer, 
            'results/test_results/confusion_matrix.png'
        )
        
    except Exception as e:
        logger.error(f"Combined testing failed: {str(e)}")
        raise




# Main function to load config and start training
def main():
    try:
        # Load config
        with open('configs/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create base output directories
        os.makedirs(config['testing']['results_path'], exist_ok=True)
        os.makedirs(config['testing']['logs_path'], exist_ok=True)
        
        if config['testing']['inference_scheme'] == 'individual':
            test_individual(config)
        else:
            test_combined(config)
            
    except Exception as e:
        logger.error(f"Testing failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
