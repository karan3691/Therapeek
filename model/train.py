import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
from tqdm import tqdm
import gc
import psutil

# Create model directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def optimize_memory():
    """
    Optimize memory usage for machines with limited RAM (8GB).
    Implements aggressive memory management techniques for MacBooks.
    """
    # Force garbage collection with generational collection
    gc.collect(2)  # Full collection including oldest generation
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Print memory usage information
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {current_memory:.2f} MB")
    
    # Set lower memory usage thresholds for PyTorch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.6)  # More restrictive GPU memory limit
    
    # More aggressive memory management for MacBooks with 8GB RAM
    if current_memory > 4000:  # If using more than 4GB
        print("WARNING: High memory usage detected. Performing aggressive cleanup...")
        # Force Python garbage collection across all generations
        for i in range(3):
            gc.collect(i)
        
        # Explicitly clear caches if available
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # If still using too much memory, take more drastic measures
        if process.memory_info().rss / 1024 / 1024 > 5500:  # If using more than 5.5GB
            print("CRITICAL: Memory usage extremely high. Emergency cleanup initiated.")
            import time
            time.sleep(2)  # Pause to allow memory to be freed
            gc.collect(2)

def load_dataset(file_path, chunk_size=None):
    """
    Load and prepare the dataset for training.
    If chunk_size is provided, load the dataset in chunks to save memory.
    """
    print(f"Loading dataset from {file_path}...")
    
    if chunk_size:
        # Load in chunks to save memory
        chunks = []
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Loading chunks"):
            # Verify the dataset structure
            if 'input' not in chunk.columns or 'response' not in chunk.columns:
                raise ValueError(f"Dataset missing required columns. Found: {chunk.columns.tolist()}")
            chunks.append(chunk)
        
        # Combine chunks
        df = pd.concat(chunks)
        
        # Free memory
        del chunks
        optimize_memory()
    else:
        # Load entire dataset at once
        df = pd.read_csv(file_path)
        
        # Verify the dataset structure
        if 'input' not in df.columns or 'response' not in df.columns:
            raise ValueError(f"Dataset missing required columns. Found: {df.columns.tolist()}")
    
    return df

def prepare_dataset_for_dialogpt(df, tokenizer, max_length=256, batch_size=1000):
    """
    Prepare the dataset for DialoGPT fine-tuning with improved preprocessing.
    Process in batches to reduce memory usage.
    """
    print("Preparing dataset for DialoGPT...")
    
    # Clean and validate the data
    df = df.dropna()  # Remove any rows with missing values
    df = df[df['input'].str.len() > 10]  # Remove very short inputs
    df = df[df['response'].str.len() > 10]  # Remove very short responses
    
    # Format conversations for DialoGPT with context management
    formatted_conversations = []
    context_window = []
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            # Add current conversation to context window
            context_window.append((row['input'], row['response']))
            if len(context_window) > 3:  # Keep last 3 turns for context
                context_window.pop(0)
            
            # Format conversation with context
            conversation = ""
            for prev_input, prev_response in context_window[:-1]:
                conversation += f"{prev_input} {tokenizer.eos_token} {prev_response} {tokenizer.eos_token} "
            conversation += f"{context_window[-1][0]} {tokenizer.eos_token} {context_window[-1][1]} {tokenizer.eos_token}"
            
            formatted_conversations.append(conversation)
        
        # Periodically free memory
        if (i // batch_size) % 5 == 0:
            optimize_memory()
    
    # Tokenize the conversations with improved handling
    tokenized_inputs = []
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(formatted_conversations), batch_size), desc="Tokenizing batches"):
        batch_conversations = formatted_conversations[i:i+batch_size]
        
        for conversation in batch_conversations:
            # Tokenize with dynamic padding and truncation
            inputs = tokenizer(
                conversation,
                truncation=True,
                max_length=max_length,
                padding='max_length',
                return_tensors="pt",
                return_special_tokens_mask=True
            )
            
            # Add position IDs and token type IDs
            tokenized_inputs.append({
                'input_ids': inputs['input_ids'][0],
                'attention_mask': inputs['attention_mask'][0],
                'special_tokens_mask': inputs['special_tokens_mask'][0],
                'position_ids': torch.arange(len(inputs['input_ids'][0]))
            })
        
        # Periodically free memory
        if (i // batch_size) % 5 == 0:
            optimize_memory()
    
    # Free memory before creating Dataset object
    del formatted_conversations
    optimize_memory()
    
    # Create a Dataset object
    dataset = Dataset.from_list(tokenized_inputs)
    
    return dataset

def train_dialogpt_model(train_dataset, val_dataset, output_dir, batch_size=2, epochs=3, gradient_accumulation_steps=4, learning_rate=5e-5):
    """
    Fine-tune the DialoGPT model on the prepared dataset.
    Uses gradient accumulation to simulate larger batch sizes while using less memory.
    Implements regularization techniques to prevent overfitting.
    """
    print("Initializing DialoGPT model...")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
    
    # Apply dropout for regularization
    model.config.resid_pdrop = 0.3  # Increased from 0.2 for stronger regularization
    model.config.embd_pdrop = 0.3  # Increased from 0.2 for stronger regularization
    model.config.attn_pdrop = 0.3  # Increased from 0.2 for stronger regularization
    
    # Set up training arguments with memory optimization and regularization
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,  # Accumulate gradients to simulate larger batch size
        learning_rate=learning_rate,  # Control learning rate to prevent overfitting
        weight_decay=0.01,  # L2 regularization to prevent overfitting
        eval_steps=200,  # Evaluate more frequently
        save_steps=500,
        save_total_limit=2,  # Only keep the 2 most recent checkpoints to save disk space
        warmup_steps=200,
        evaluation_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU is available
        dataloader_num_workers=1,  # Reduce number of workers to save memory
        dataloader_pin_memory=False,  # Disable pin memory to reduce memory usage
        early_stopping_patience=3,  # Stop training if no improvement after 3 evaluations
        early_stopping_threshold=0.01  # Minimum improvement required
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False  # We're not using masked language modeling
    )
    
    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Free memory after training
    optimize_memory()
    
    # Save the model and tokenizer to output directory
    model_path = os.path.join(output_dir, "fine_tuned_dialogpt")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    # Also save to the model directory for easier access by other scripts
    model_dir = os.path.dirname(os.path.abspath(__file__))
    direct_model_path = os.path.join(model_dir, "fine_tuned_dialogpt")
    
    # Create the directory if it doesn't exist
    os.makedirs(direct_model_path, exist_ok=True)
    
    # Save model and tokenizer to the direct path
    model.save_pretrained(direct_model_path)
    tokenizer.save_pretrained(direct_model_path)
    
    print(f"Model and tokenizer saved to {model_path} and {direct_model_path}")
    
    return model, tokenizer

def evaluate_model(model, tokenizer, test_dataset, num_samples=5):
    """
    Evaluate the fine-tuned model on a subset of the test dataset.
    Reduced number of samples and optimized generation parameters for faster response time.
    """
    print("\nEvaluating model on sample conversations...")
    
    # Set the model to evaluation mode
    model.eval()
    
    # Get a subset of the test dataset for evaluation
    test_samples = test_dataset.shuffle().select(range(min(num_samples, len(test_dataset))))
    
    for i, sample in enumerate(test_samples):
        # Get the input_ids and convert to text
        input_text = tokenizer.decode(sample['input_ids'], skip_special_tokens=True)
        
        # Generate a response
        input_ids = torch.tensor([sample['input_ids']]).to(model.device)
        attention_mask = torch.tensor([sample['attention_mask']]).to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=len(input_ids[0]) + 30,  # Reduced max length for faster generation
                pad_token_id=tokenizer.eos_token_id,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                num_beams=1,  # Use greedy decoding for faster generation
                early_stopping=True  # Stop generation when all beams reach EOS
            )
        
        # Decode the generated output
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Print the results
        print(f"\nSample {i+1}:")
        print(f"Input: {input_text}")
        print(f"Generated: {generated_text}")
        
        # Free memory after each sample
        optimize_memory()

if __name__ == "__main__":
    # Define file paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(model_dir), "data")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    output_dir = os.path.join(model_dir, "output")
    
    # Check if required files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}. Please run preprocess.py first.")
            exit(1)
    
    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare datasets with chunking for memory efficiency
    chunk_size = 5000  # Process 5000 rows at a time
    train_df = load_dataset(train_path, chunk_size=chunk_size)
    val_df = load_dataset(val_path, chunk_size=chunk_size)
    test_df = load_dataset(test_path, chunk_size=chunk_size)
    
    # Prepare datasets for DialoGPT with batch processing
    batch_size = 1000  # Process 1000 conversations at a time
    train_dataset = prepare_dataset_for_dialogpt(train_df, tokenizer, batch_size=batch_size)
    val_dataset = prepare_dataset_for_dialogpt(val_df, tokenizer, batch_size=batch_size)
    test_dataset = prepare_dataset_for_dialogpt(test_df, tokenizer, batch_size=batch_size)
    
    # Free up memory after dataset preparation
    del train_df, val_df, test_df
    optimize_memory()
    
    # Train the model with memory-efficient settings
    model, tokenizer = train_dialogpt_model(
        train_dataset, 
        val_dataset, 
        output_dir,
        batch_size=4,  # Increased from 1 to improve generalization
        epochs=5,      # Increased epochs for better learning
        gradient_accumulation_steps=4,  # Reduced from 8 to match new batch size
        learning_rate=5e-6  # Reduced learning rate for better generalization
    )
    
    # Evaluate the model
    evaluate_model(model, tokenizer, test_dataset, num_samples=5)  # Reduced number of samples
    
    print("\nModel training and evaluation complete!")
    print("Next step: Run 'python model/quantize.py' to optimize the model for inference")