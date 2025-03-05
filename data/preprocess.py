import os
import pandas as pd
import numpy as np
import re
import nltk
import gc
from nltk.tokenize import word_tokenize
from tqdm import tqdm

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """
    Clean and normalize text data.
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters and numbers but keep emoticons and important punctuation
    text = re.sub(r'[^\w\s\?\!\.,:\)\(\-]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess_mental_health_dataset(file_path):
    """
    Preprocess the Mental Health Conversational dataset.
    """
    print(f"Preprocessing {file_path}...")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['question', 'answer']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Dataset missing required columns. Found: {df.columns.tolist()}")
            return None
        
        # Clean text
        print("Cleaning text data...")
        df['clean_question'] = df['question'].apply(clean_text)
        df['clean_answer'] = df['answer'].apply(clean_text)
        
        # Remove rows with empty questions or answers
        df = df[df['clean_question'].str.len() > 0]
        df = df[df['clean_answer'].str.len() > 0]
        
        # Create conversation pairs
        conversations = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating conversation pairs"):
            conversations.append({
                'input': row['clean_question'],
                'response': row['clean_answer']
            })
        
        print(f"Processed {len(conversations)} conversation pairs")
        return conversations
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None

def preprocess_human_conversation_dataset(file_path):
    """
    Preprocess the Human Conversation dataset.
    """
    print(f"Preprocessing {file_path}...")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['input', 'response']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Dataset missing required columns. Found: {df.columns.tolist()}")
            return None
        
        # Clean text
        print("Cleaning text data...")
        df['clean_input'] = df['input'].apply(clean_text)
        df['clean_response'] = df['response'].apply(clean_text)
        
        # Remove rows with empty inputs or responses
        df = df[df['clean_input'].str.len() > 0]
        df = df[df['clean_response'].str.len() > 0]
        
        # Create conversation pairs
        conversations = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating conversation pairs"):
            conversations.append({
                'input': row['clean_input'],
                'response': row['clean_response']
            })
        
        print(f"Processed {len(conversations)} conversation pairs")
        return conversations
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None

def preprocess_reddit_dataset(file_path):
    """
    Preprocess the Reddit Mental Health Support dataset.
    """
    print(f"Preprocessing {file_path}...")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check columns and adapt based on actual structure
        if 'post' in df.columns and 'comment' in df.columns:
            input_col, response_col = 'post', 'comment'
        elif 'title' in df.columns and 'selftext' in df.columns and 'comment' in df.columns:
            # Combine title and selftext for input
            df['input'] = df['title'] + " " + df['selftext'].fillna('')
            input_col, response_col = 'input', 'comment'
        else:
            print(f"Error: Unexpected column structure. Found: {df.columns.tolist()}")
            return None
        
        # Clean text
        print("Cleaning text data...")
        df['clean_input'] = df[input_col].apply(clean_text)
        df['clean_response'] = df[response_col].apply(clean_text)
        
        # Remove rows with empty inputs or responses
        df = df[df['clean_input'].str.len() > 0]
        df = df[df['clean_response'].str.len() > 0]
        
        # Create conversation pairs
        conversations = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating conversation pairs"):
            conversations.append({
                'input': row['clean_input'],
                'response': row['clean_response']
            })
        
        print(f"Processed {len(conversations)} conversation pairs")
        return conversations
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None

def preprocess_genz_dataset(file_path):
    """
    Preprocess the Gen Z conversation dataset.
    """
    print(f"Preprocessing {file_path}...")
    
    try:
        # Load dataset
        df = pd.read_csv(file_path)
        
        # Check if required columns exist
        required_columns = ['input', 'response']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: Dataset missing required columns. Found: {df.columns.tolist()}")
            return None
        
        # Clean text
        print("Cleaning text data...")
        df['clean_input'] = df['input'].apply(clean_text)
        df['clean_response'] = df['response'].apply(clean_text)
        
        # Remove rows with empty inputs or responses
        df = df[df['clean_input'].str.len() > 0]
        df = df[df['clean_response'].str.len() > 0]
        
        # Create conversation pairs
        conversations = []
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating conversation pairs"):
            conversations.append({
                'input': row['clean_input'],
                'response': row['clean_response']
            })
        
        print(f"Processed {len(conversations)} conversation pairs")
        return conversations
    
    except Exception as e:
        print(f"Error preprocessing dataset: {e}")
        return None

def combine_datasets(datasets):
    """
    Combine multiple preprocessed datasets into a single dataset.
    Uses batch processing for memory efficiency.
    """
    combined = []
    batch_size = 5000  # Process 5000 conversations at a time
    
    for dataset in datasets:
        if dataset:
            for i in range(0, len(dataset), batch_size):
                batch = dataset[i:i+batch_size]
                combined.extend(batch)
                
                # Free memory periodically
                if i % (batch_size * 5) == 0:
                    gc.collect()
    
    print(f"Combined dataset contains {len(combined)} conversation pairs")
    return combined

def save_processed_data(data, output_path):
    """
    Save processed data to a CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")

def create_train_val_test_split(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, is_genz=False):
    """
    Split the dataset into training, validation, and test sets.
    Uses batch processing for memory efficiency.
    For Gen Z dataset, ensures proportional distribution across all splits.
    """
    # Validate ratios
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")
    
    # Calculate split indices
    total_size = len(data)
    train_end = int(total_size * train_ratio)
    val_end = train_end + int(total_size * val_ratio)
    
    # Process in batches
    batch_size = 5000
    train_data, val_data, test_data = [], [], []
    
    # For Gen Z dataset, ensure proportional distribution
    if is_genz:
        # Shuffle data to ensure random distribution while maintaining proportions
        indices = list(range(total_size))
        np.random.shuffle(indices)
        
        # Calculate target sizes for each split
        train_size = int(total_size * train_ratio)
        val_size = int(total_size * val_ratio)
        
        for i in tqdm(range(0, total_size, batch_size), desc="Splitting Gen Z dataset"):
            batch_indices = indices[i:i+batch_size]
            batch = [data[idx] for idx in batch_indices]
            
            for j, item in enumerate(batch):
                if len(train_data) < train_size:
                    train_data.append(item)
                elif len(val_data) < val_size:
                    val_data.append(item)
                else:
                    test_data.append(item)
            
            # Free memory periodically
            if i % (batch_size * 5) == 0:
                gc.collect()
    else:
        # Original splitting logic for non-Gen Z datasets
        for i in tqdm(range(0, total_size, batch_size), desc="Splitting dataset"):
            batch = data[i:i+batch_size]
            batch_start = i
            batch_end = i + len(batch)
            
            if batch_end <= train_end:
                train_data.extend(batch)
            elif batch_start >= train_end and batch_end <= val_end:
                val_data.extend(batch)
            elif batch_start >= val_end:
                test_data.extend(batch)
            else:
                for j, item in enumerate(batch):
                    if i + j < train_end:
                        train_data.append(item)
                    elif i + j < val_end:
                        val_data.append(item)
                    else:
                        test_data.append(item)
            
            # Free memory periodically
            if i % (batch_size * 5) == 0:
                gc.collect()
    
    print(f"Data split: Train={len(train_data)}, Validation={len(val_data)}, Test={len(test_data)}")
    return train_data, val_data, test_data

if __name__ == "__main__":
    # Define file paths
    data_dir = os.path.dirname(os.path.abspath(__file__))
    mental_health_path = os.path.join(data_dir, "mental_health_conversational.csv")
    reddit_path = os.path.join(data_dir, "reddit_mental_health_support.csv")
    genz_path = os.path.join(data_dir, "genz_conversations_processed.csv")
    
    # Check if at least one dataset exists
    if not os.path.exists(mental_health_path) and not os.path.exists(reddit_path) and not os.path.exists(genz_path):
        print("Error: No datasets found. Please run download_dataset.py and download_genz_dataset.py first.")
        exit(1)
    
    # Preprocess datasets
    mental_health_data = None
    reddit_data = None
    genz_data = None
    
    if os.path.exists(mental_health_path):
        mental_health_data = preprocess_mental_health_dataset(mental_health_path)
    
    if os.path.exists(reddit_path):
        reddit_data = preprocess_reddit_dataset(reddit_path)
    
    if os.path.exists(genz_path):
        genz_data = preprocess_genz_dataset(genz_path)
    
    # Combine datasets
    combined_data = combine_datasets([mental_health_data, reddit_data, genz_data])
    
    if combined_data:
        # Create data splits with proportional Gen Z distribution
        train_data, val_data, test_data = create_train_val_test_split(combined_data, is_genz=True if genz_data else False)
        
        # Save processed data
        save_processed_data(train_data, os.path.join(data_dir, "train.csv"))
        save_processed_data(val_data, os.path.join(data_dir, "val.csv"))
        save_processed_data(test_data, os.path.join(data_dir, "test.csv"))
        
        print("\nData preprocessing complete!")
        print("Next step: Run 'python model/train.py' to fine-tune the DialoGPT model")
    else:
        print("\nError: Failed to preprocess datasets")