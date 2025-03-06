import os
import pandas as pd
import numpy as np
import re
import nltk
import gc
import torch
import psutil
import platform
from tqdm import tqdm

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def optimize_memory():
    """Optimize memory usage for machines with limited RAM (8GB)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {current_memory:.2f} MB")
    
    if current_memory > 3500:
        print("WARNING: High memory usage detected. Performing aggressive cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

def clean_text(text):
    """
    Cleans and normalizes text data.
    """
    if not isinstance(text, str):
        return ""
    
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'[^\w\s\?\!\.,:\)\(\-]', '', text)  # Remove special characters (but keep punctuation)
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra whitespace
    
    return text

def preprocess_dataset(file_path, input_col, response_col):
    """
    Generic function to preprocess any dataset.
    Handles different column name mappings for various datasets.
    """
    if not os.path.exists(file_path):
        print(f"Skipping: {file_path} not found.")
        return None
    
    print(f"Preprocessing {file_path}...")
    
    try:
        df = pd.read_csv(file_path)
        
        # Define column mappings for different datasets
        column_mappings = {
            'text': ('text', None),  # For mental_health_conversational.csv
            'Context': ('Context', 'Response'),  # For mental_health_counseling_conversations.csv
            'context': ('context', 'response'),  # For mental_health_counseling_rated.csv
            'question': ('question', 'response_j')  # For psychology_dataset.csv
        }
        
        # Detect and apply appropriate column mapping
        applied_mapping = None
        for key, (in_col, out_col) in column_mappings.items():
            if key in df.columns:
                if key == 'text':
                    # Special handling for text column that contains both input and response
                    df['input'] = df['text'].apply(lambda x: x.split('<ASSISTANT>:')[0].replace('<HUMAN>:', '').strip() if isinstance(x, str) else '')
                    df['response'] = df['text'].apply(lambda x: x.split('<ASSISTANT>:')[1].strip() if isinstance(x, str) and '<ASSISTANT>:' in x else '')
                else:
                    df['input'] = df[in_col]
                    df['response'] = df[out_col] if out_col else ''
                applied_mapping = (in_col, out_col)
                break
        
        if applied_mapping is None:
            print(f"Error: {file_path} has unsupported column format. Found: {df.columns.tolist()}")
            return None
        
        df['clean_input'] = df[input_col].apply(clean_text)
        df['clean_response'] = df[response_col].apply(clean_text)
        
        df = df[df['clean_input'].str.len() > 0]
        df = df[df['clean_response'].str.len() > 0]
        
        conversations = [
            {'input': row['clean_input'], 'response': row['clean_response']}
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {file_path}")
        ]
        
        print(f"Processed {len(conversations)} conversation pairs from {file_path}")
        return conversations
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def combine_datasets(datasets):
    """
    Combine multiple preprocessed datasets into a single dataset.
    Uses batch processing for memory efficiency.
    """
    combined = []
    batch_size = 5000  # Process in chunks

    for dataset in datasets:
        if dataset:
            for i in range(0, len(dataset), batch_size):
                combined.extend(dataset[i:i+batch_size])
                gc.collect()

    print(f"Final dataset contains {len(combined)} conversation pairs")
    return combined

def save_processed_data(data, output_path):
    """
    Save processed data to a CSV file.
    """
    if not data:
        print(f"Skipping save: No data for {output_path}")
        return
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split the dataset into training, validation, and test sets.
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    
    np.random.shuffle(data)
    
    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)
    
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]
    
    print(f"Data split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

if __name__ == "__main__":
    print("\n🔄 Starting Preprocessing...\n")

    # Define file paths from `download_dataset.py`
    data_dir = os.path.dirname(os.path.abspath(__file__))
    data_subdir = os.path.join(data_dir, "data")
    dataset_paths = {
        "mental_health_conversational": os.path.join(data_subdir, "mental_health_conversational.csv"),
        "mental_health_counseling": os.path.join(data_subdir, "mental_health_counseling_conversations.csv"),
        "mental_health_rated": os.path.join(data_subdir, "mental_health_counseling_rated.csv"),
        "psychology": os.path.join(data_subdir, "psychology_dataset.csv")
    }

    # Preprocess all datasets
    datasets = {
        "mental_health_conversational": preprocess_dataset(dataset_paths["mental_health_conversational"], "text", None),
        "mental_health_counseling": preprocess_dataset(dataset_paths["mental_health_counseling"], "Context", "Response"),
        "mental_health_rated": preprocess_dataset(dataset_paths["mental_health_rated"], "context", "response"),
        "psychology": preprocess_dataset(dataset_paths["psychology"], "question", "response_j")
    }

    # Combine datasets
    combined_data = combine_datasets(list(datasets.values()))

    if combined_data:
        # Split dataset into train/val/test
        train_data, val_data, test_data = create_train_val_test_split(combined_data)

        # Save processed datasets
        save_processed_data(train_data, os.path.join(data_dir, "train.csv"))
        save_processed_data(val_data, os.path.join(data_dir, "val.csv"))
        save_processed_data(test_data, os.path.join(data_dir, "test.csv"))

        print("\n✅ Data preprocessing complete!")
        print("Next step: Run `python model/finetune_zephyr.py` to fine-tune Zephyr-7B Beta.")
    else:
        print("\n❌ Error: No datasets processed. Please check `download_dataset.py` output.")
