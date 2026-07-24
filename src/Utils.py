import pandas as pd
import re
from urllib.parse import unquote, urlparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.utils import resample
from typing import List
import os
import glob
from transformers import TrainingArguments, AutoTokenizer

nltk.download('stopwords')
nltk.download('punkt')


def replace_url(match):
    url = match.group(0)
    parsed_url = urlparse(url)
    domain = parsed_url.netloc.replace('www.', '').replace('.com', '').replace('.org', '').replace('.net', '')
    path = unquote(parsed_url.path)

    tokens = re.split(r'[/@]', path)
    tokens = [t for t in tokens if t and len(t) < 20 and not re.match(r'^[0-9a-f]{16,}$', t, re.IGNORECASE)]

    if '@' in url:
        email_domain = re.findall(r'@<*(.*?)>*$', url)
        if email_domain:
            tokens += email_domain[0].split('.')

    tokens = list(dict.fromkeys(tokens))
    tokens = [re.sub(r'[^a-z]', ' ', token.lower()) for token in tokens if token.isalnum() or token.isalpha()]
    tokens = [token for token in tokens if token]

    return 'url of ' + ' '.join([domain] + tokens)

def preprocess_bug_report(text):
    if pd.isnull(text):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Format URLs
    text = re.sub(r'https?://[^\s]+', replace_url, text)

    # 3. Normalize config keys and code-like tokens
    text = re.sub(r'\{\{([^}]+)\}\}', r'\1', text)  # {{config.property}} → config.property
    text = re.sub(r'\b[a-zA-Z_]+\.[a-zA-Z0-9_.]+\b', lambda m: m.group(0).replace('.', ' '), text)  # keep config.package.class names as token sequences

    # 4. Remove punctuation except for useful symbols in code (e.g. underscore, parentheses, dot)
    allowed = set('_().')
    text = ''.join(ch if ch.isalnum() or ch.isspace() or ch in allowed else ' ' for ch in text)

    # 5. Tokenize and remove stopwords (but keep code-relevant terms)
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    tokens = [tok for tok in tokens if tok not in stop_words and (tok.isalpha() or re.match(r'[a-zA-Z_]+\(\)', tok))]

    return ' '.join(tokens)


def clean_dataframe(df, text_col="text", label_col="label", allowed_classes=[0, 1]):
    df[text_col] = df[text_col].apply(lambda x: x if pd.notnull(x) and str(x).strip() != "" else None)
    df.dropna(subset=[text_col, label_col], inplace=True)
    df = df[df[label_col].isin(allowed_classes)]
    return df

def balance_dataset(df):
    
    print(f"Total Samples: {len(df)}")
    print("Class Distribution:\n", df['label'].value_counts())

    # Separate majority and minority classes
    df_majority = df[df.label == 0]
    df_minority = df[df.label == 1]

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,               # sample with replacement
        n_samples=len(df_majority), # match majority count
        random_state=42
    )

    # Combine majority and upsampled minority
    df_balanced = pd.concat([df_majority, df_minority_upsampled]).sample(frac=1).reset_index(drop=True)


    print(f"\n Total Samples of balanced data: {len(df_balanced)}")
    print("\n Class Distribution of balanced:\n", df_balanced['label'].value_counts())
    return df_balanced

def load_and_preprocess_dataset(file_path: str, logger) -> pd.DataFrame:
    """Load and preprocess a single dataset file."""
    try:
        logger.info(f"Loading dataset: {file_path}")
        df = pd.read_csv(file_path, names=['text', 'label'])
        df = clean_dataframe(df)
        df['text'] = df['text'].apply(preprocess_bug_report)
        df = balance_dataset(df)
        return df
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        raise

def get_dataset_files(data_dir: str, logger) -> List[str]:
    """Get all CSV files from the data directory."""
    try:
        files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not files:
            raise FileNotFoundError(f"No CSV files found in {data_dir}")
        return files
    except Exception as e:
        logger.error(f"Error accessing data directory: {str(e)}")
        raise
    
def load_tokenizer(model_name: str):
    """Load a tokenizer with fallbacks for legacy checkpoints.

    Some code checkpoints (e.g. Salesforce/codet5-base) ship a large
    additional_special_tokens list in an old format that newer `tokenizers`
    versions reject ("Input must be a List[Union[str, AddedToken]]"). Those
    sentinel tokens are not needed for classification, so we retry without them.
    """
    last_err = None
    for kwargs in ({}, {"additional_special_tokens": []},
                   {"use_fast": False, "additional_special_tokens": []}):
        try:
            return AutoTokenizer.from_pretrained(model_name, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"Could not load tokenizer for {model_name}: {last_err}")


def get_training_args(config: dict, prog_dir, precision: str = None,
                      batch_size: int = None) -> TrainingArguments:
    """Build TrainingArguments from config.

    precision: 'fp16' or 'bf16' (per-model; CodeT5 needs bf16/fp32 to avoid NaNs).
    batch_size: optional per-device batch-size override (e.g. smaller for CodeT5).
    """
    t = config['training']
    try:
        train_bs = batch_size or t['per_device_train_batch_size']
        eval_bs = batch_size or t['per_device_eval_batch_size']
        training_args = TrainingArguments(
                output_dir=prog_dir or t.get('output_dir', './results'),
                eval_strategy="epoch",
                save_strategy="epoch",
                learning_rate=t['learning_rate'],
                per_device_train_batch_size=train_bs,
                per_device_eval_batch_size=eval_bs,
                num_train_epochs=t['num_train_epochs'],
                weight_decay=t['weight_decay'],
                label_smoothing_factor=t['label_smoothing_factor'],
                max_grad_norm=t['max_grad_norm'],
                lr_scheduler_type=t['lr_scheduler_type'],
                warmup_steps=t['warmup_steps'],
                logging_dir=os.path.join('results', 'logs'),
                logging_steps=t['logging_steps'],
                load_best_model_at_end=t['load_best_model_at_end'],
                metric_for_best_model=t['metric_for_best_model'],
                greater_is_better=t['greater_is_better'],
                fp16=(precision == 'fp16'),
                bf16=(precision == 'bf16'),
                remove_unused_columns=t.get('remove_unused_columns', False),
                dataloader_drop_last=t.get('dataloader_drop_last', False),
                overwrite_output_dir=t.get('overwrite_output_dir', True),
                report_to="none",
            )
        return training_args
    except KeyError as e:
        raise KeyError(f"Missing key in config: {str(e)}")
