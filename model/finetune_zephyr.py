import os
import torch
import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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

def prepare_dataset_for_zephyr(df, tokenizer, max_length=512, batch_size=1000):
    """
    Prepare the dataset for Zephyr-3B fine-tuning with improved preprocessing.
    Process in batches to reduce memory usage.
    """
    print("Preparing dataset for Zephyr-3B...")
    
    # Clean and validate the data
    df = df.dropna()  # Remove any rows with missing values
    df = df[df['input'].str.len() > 10]  # Remove very short inputs
    df = df[df['response'].str.len() > 10]  # Remove very short responses
    
    # Format conversations for Zephyr with context management
    formatted_data = []
    
    # Process in batches to save memory
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            # Format conversation for Zephyr
            # Using the Zephyr chat template format: <|user|>\n{query}<|assistant|>\n{response}
            conversation = f"<|user|>\n{row['input']}<|assistant|>\n{row['response']}"
            
            # Tokenize with dynamic padding and truncation
            inputs = tokenizer(conversation, truncation=True, max_length=max_length)
            
            formatted_data.append({
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"]
            })
        
        # Periodically free memory
        if (i // batch_size) % 5 == 0:
            optimize_memory()
    
    # Create a Dataset object
    dataset = Dataset.from_list(formatted_data)
    
    return dataset

def setup_qlora_config():
    """
    Set up QLoRA configuration for memory-efficient fine-tuning.
    """
    return LoraConfig(
        r=16,  # Rank dimension
        lora_alpha=32,  # Alpha parameter for LoRA scaling
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def train_zephyr_model(train_dataset, val_dataset, output_dir, batch_size=1, epochs=3, gradient_accumulation_steps=16):
    """
    Fine-tune the Zephyr-3B model using QLoRA for memory efficiency.
    """
    print("Initializing Zephyr-3B model with 4-bit quantization...")
    
    # Set up quantization configuration for 4-bit precision
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # Load the base model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        "HuggingFaceH4/zephyr-7b-beta",  # Using the 7B model as base, will be quantized to fit in 8GB RAM
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Apply LoRA
    lora_config = setup_qlora_config()
    model = get_peft_model(model, lora_config)
    
    # Print trainable parameters info
    print(f"Trainable parameters: {model.print_trainable_parameters()}")
    
    # Set up training arguments with memory optimization
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=16
    )
    
    # Convert the fine-tuned model to GGUF format for use with llama.cpp
    print("\nConverting model to GGUF format...")
    gguf_success = convert_to_gguf(
        os.path.join(output_dir, "zephyr_3b_finetuned"),
        gguf_output_path
    )
    
    if gguf_success:
        # Evaluate the model
        evaluate_model(gguf_output_path, test_dataset, num_samples=5)
    else:
        print("Error: GGUF conversion failed. Skipping evaluation.")
    
    print("\nModel training, conversion, and evaluation complete!")
    print("Next step: Run 'python api/app.py' to start the FastAPI server")

    # Initialize the Trainer
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Free memory after training
    optimize_memory()
    
    # Save the model and tokenizer
    model_path = os.path.join(output_dir, "zephyr_3b_finetuned")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    
    print(f"Model and tokenizer saved to {model_path}")
    
    return model, tokenizer

def convert_to_gguf(model_path, output_path):
    """
    Convert the fine-tuned model to GGUF format for use with llama.cpp.
    """
    print(f"Converting model to GGUF format...")
    
    try:
        # Use llama.cpp's conversion script
        import subprocess
        cmd = [
            "python", "-m", "llama_cpp.convert_hf_to_gguf",
            model_path,
            "--outfile", output_path,
            "--outtype", "q4_0"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted model to GGUF format at {output_path}")
            return True
        else:
            print(f"Error converting model: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception during model conversion: {str(e)}")
        return False

def evaluate_model(model_path, test_dataset, num_samples=5):
    """
    Evaluate the fine-tuned model on a subset of the test dataset.
    """
    print("\nEvaluating model on sample conversations...")
    
    # Load the model and tokenizer
    from llama_cpp import Llama
    
    # Load the model using llama.cpp
    model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
    
    # Get a subset of the test dataset for evaluation
    test_samples = test_dataset.shuffle().select(range(min(num_samples, len(test_dataset))))
    
    for i, sample in enumerate(test_samples):
        # Extract the user message from the input
        input_text = sample["input_ids"]
        
        # Generate a response
        prompt = f"<|user|>\n{input_text}<|assistant|>\n"
        response = model(prompt, max_tokens=150, temperature=0.7, top_p=0.9, top_k=40)
        
        # Print the results
        print(f"\nSample {i+1}:")
        print(f"Input: {input_text}")
        print(f"Generated: {response['choices'][0]['text']}")
        
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
    gguf_output_path = os.path.join(model_dir, "zephyr-3b-q4.gguf")
    
    # Check if required files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}. Please run preprocess.py first.")
            exit(1)
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and prepare datasets with chunking for memory efficiency
    chunk_size = 5000  # Process 5000 rows at a time
    train_df = load_dataset(train_path, chunk_size=chunk_size)
    val_df = load_dataset(val_path, chunk_size=chunk_size)
    test_df = load_dataset(test_path, chunk_size=chunk_size)
    
    # Prepare datasets for Zephyr with batch processing
    batch_size = 1000  # Process 1000 conversations at a time
    train_dataset = prepare_dataset_for_zephyr(train_df, tokenizer, batch_size=batch_size)
    val_dataset = prepare_dataset_for_zephyr(val_df, tokenizer, batch_size=batch_size)
    test_dataset = prepare_dataset_for_zephyr(test_df, tokenizer, batch_size=batch_size)
    
    # Free up memory after dataset preparation
    del train_df, val_df, test_df
    optimize_memory()
    
    # Train the model with memory-efficient settings
    model, tokenizer = train_zephyr_model(
        train_dataset, 
        val_dataset, 
        output_dir,
        batch_size=1,
        epochs=3,