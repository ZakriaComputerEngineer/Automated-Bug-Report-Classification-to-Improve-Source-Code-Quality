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


# trainning on multiple datasets
# in a progressive manner
def train_Continual(config: dict):
    """Train the model progressively on multiple datasets."""
    try:
        dataset_files = get_dataset_files(config['data']['data_dir'], logger)
        
        model = None
        
        for file_path in dataset_files:
            
            dataset_name = Path(file_path).stem
            logger.info(f"\nStarting training on dataset: {dataset_name}")
            
            # Create dataset-specific output directories
            prog_dir = os.path.join(config['data']['progressive_save_dir'], dataset_name)
            os.makedirs(prog_dir, exist_ok=True)
            
            # Load and preprocess current dataset
            df = load_and_preprocess_dataset(file_path, logger)
            train_dataset, val_dataset = create_datasets(df, config, logger)
            
            # Initialize or load model
            if model is None:
                model = AutoModelForSequenceClassification.from_pretrained(
                    config['model']['name'],
                    num_labels=config['model']['num_labels']
                )
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model.to(device)
            
            training_args = get_training_args(config, prog_dir)
            
            # Initialize Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics
            )
            
            # Train
            trainer.train()
            
            # Save model and evaluation results
            trainer.evaluate()
            
            # Save the model
            trainer.save_model(os.path.join(prog_dir, 'model'))
            
            # Save evaluation results
            plot_confusion_matrix(
                val_dataset, 
                trainer, 
                os.path.join(prog_dir, 'confusion_matrix.png')
            )
            
            # Update model for next dataset
            model = trainer.model
            
    except Exception as e:
        logger.error(f"Continual training failed: {str(e)}")
        raise





# trainning on all datasets combined
def train_combined(config: dict):
    """Train on all datasets combined."""
    try:
        dataset_files = get_dataset_files(config['data']['data_dir'], logger)
        
        # Combine all datasets
        dfs = []
        for file_path in dataset_files:
            df = load_and_preprocess_dataset(file_path, logger)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True).sample(frac=1).reset_index(drop=True)
        
        train_dataset, val_dataset = create_datasets(combined_df, config, logger)
        
        # Initialize model and training
        model = AutoModelForSequenceClassification.from_pretrained(
            config['model']['name'],
            num_labels=config['model']['num_labels']
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        training_args = get_training_args(config, None)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics
        )
        
        # Train the model
        trainer.train()
        
        # Evaluate the model
        trainer.evaluate()
        
        # Save the final model
        trainer.save_model('results/models/final_model')
        
        # Save evaluation results
        plot_confusion_matrix(
            val_dataset, 
            trainer, 
            'results/plots/confusion_matrix.png'
        )
        
    except Exception as e:
        logger.error(f"Combined training failed: {str(e)}")
        raise




# Main function to load config and start training
def main():
    try:
        # Load config
        with open('configs/model_config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Create base output directories
        os.makedirs(config["training"]['results_models'], exist_ok=True)
        os.makedirs(config["training"]['results_plots'], exist_ok=True)
        os.makedirs(config["training"]['results_logs'], exist_ok=True)
        
        if config['training']['learning'] == 'Continual':
            os.makedirs(config['training']['progressive_save_dir'], exist_ok=True)
            train_Continual(config)
        else:
            train_combined(config)
            
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
