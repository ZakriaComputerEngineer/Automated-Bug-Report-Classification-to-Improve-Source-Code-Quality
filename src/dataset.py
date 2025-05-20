import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from typing import List, Tuple

class BugReportDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item



def create_datasets(df: pd.DataFrame, config: dict, logger, task='training') -> Tuple[BugReportDataset, BugReportDataset]:
    """Create train and validation datasets from DataFrame."""
    try:
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            df["text"].tolist(),
            df["label"].tolist(),
            test_size=config['data']['train_test_split'],
            random_state=config['data']['random_seed']
        )

        if task == 'training':
            logger.info(f"Training on {len(train_texts)} samples and validating on {len(val_texts)} samples.")
            tokenizer = AutoTokenizer.from_pretrained(config['model']['name'])
        else:
            logger.info(f"Testing on {len(val_texts)} samples.")
            tokenizer = AutoTokenizer.from_pretrained(config['testing']['trained_model_path'])
        
        train_encodings = tokenizer(
            train_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=config['data']['max_length']
        )
        val_encodings = tokenizer(
            val_texts, 
            truncation=True, 
            padding="max_length", 
            max_length=config['data']['max_length']
        )

        return (
            BugReportDataset(train_encodings, train_labels),
            BugReportDataset(val_encodings, val_labels)
        )
    except Exception as e:
        logger.error(f"Error creating datasets: {str(e)}")
        raise