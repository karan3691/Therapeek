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
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm
import gc
import psutil
import subprocess
import importlib
import pkg_resources
import platform

# Create model directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Check bitsandbytes version
def check_bitsandbytes_version():
    """Check if bitsandbytes is installed and has the required version for 4-bit quantization."""
    try:
        import bitsandbytes as bnb
        bnb_version = pkg_resources.get_distribution("bitsandbytes").version
        required_version = "0.41.0"
        
        if pkg_resources.parse_version(bnb_version) < pkg_resources.parse_version(required_version):
            print(f"\nWARNING: bitsandbytes version {bnb_version} is installed, but version {required_version} or higher is required for 4-bit quantization.")
            print(f"Please upgrade bitsandbytes: pip install -U bitsandbytes>={required_version}")
            return False
        return True
    except ImportError:
        print("\nERROR: bitsandbytes is not installed. It is required for 4-bit quantization.")
        print("Please install it: pip install bitsandbytes>=0.41.0")
        return False

def optimize_memory():
    """
    Optimize memory usage for machines with limited RAM (8GB).
    Implements extremely aggressive memory management techniques for MacBooks.
    Specifically tuned for Zephyr-7B model fine-tuning.
    """
    # Force garbage collection with generational collection
    gc.collect(2)  # Full collection including oldest generation
    
    # Clear device-specific cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        # Apple Silicon specific optimizations
        if platform.system() == "Darwin" and platform.processor() == "arm":
            torch.mps.set_per_process_memory_fraction(0.4)  # More conservative for M1/M2
    
    # Print memory usage information
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {current_memory:.2f} MB")
    
    # Aggressive cleanup if memory usage is high
    if current_memory > 3000:  # Lower threshold for earlier intervention
        print("WARNING: High memory usage detected. Performing aggressive cleanup...")
        for i in range(3):
            gc.collect(i)
        
        # Device-specific cache clearing
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(0.4)  # Further restrict GPU memory
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
            if platform.system() == "Darwin" and platform.processor() == "arm":
                torch.mps.set_per_process_memory_fraction(0.3)  # Even more conservative
        
        # macOS specific memory optimization
        if platform.system() == "Darwin":
            try:
                import ctypes
                lib = ctypes.CDLL("libSystem.B.dylib")
                lib.malloc_trim(0)
            except Exception as e:
                print(f"Warning: macOS memory optimization failed: {e}")
        
        # Critical memory situation
        if current_memory > 4500:  # Lower threshold for critical action
            print("CRITICAL: Memory usage extremely high. Taking emergency measures...")
            import time
            time.sleep(5)  # Longer pause to ensure memory is freed
            gc.collect(2)
            
            # Disable gradient calculation temporarily if extremely high
            if torch.is_grad_enabled():
                print("EMERGENCY: Temporarily disabling gradient calculation")
                torch.set_grad_enabled(False)
                time.sleep(1)
                torch.set_grad_enabled(True)
                
            # Release as much memory as possible
            if platform.system() == "Darwin" and torch.backends.mps.is_available():
                # Special handling for Apple Silicon
                torch.mps.empty_cache()
                print("Performed Apple Silicon specific memory cleanup")
                
            # Reduce thread count if memory is still critical
            if current_memory > 6000:
                print("EMERGENCY: Memory critically high, reducing thread count")
                import threading
                threading.stack_size(1024 * 128)  # Reduce thread stack size
    
    # Set lower memory usage thresholds for PyTorch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)  # More restrictive GPU memory limit
    
    # Aggressive cleanup if memory usage is high
    if current_memory > 3500:  # Lower threshold for earlier intervention
        print("WARNING: High memory usage detected. Performing aggressive cleanup...")
        for i in range(3):
            gc.collect(i)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Try to release memory from Python's memory allocator
        import ctypes
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        
        if current_memory > 5000:  # If using more than 5GB
            print("CRITICAL: Memory usage extremely high. Pausing to free memory.")
            import time
            time.sleep(3)  # Longer pause to ensure memory is freed
            gc.collect(2)
            
            # Set a lower memory fraction if still high
            if torch.cuda.is_available():
                torch.cuda.set_per_process_memory_fraction(0.4)
                
            # Disable gradient calculation temporarily if extremely high
            if current_memory > 6000 and torch.is_grad_enabled():
                print("EMERGENCY: Temporarily disabling gradient calculation")
                torch.set_grad_enabled(False)
                time.sleep(1)
                torch.set_grad_enabled(True)

def load_dataset(file_path, chunk_size=5000):
    """
    Load and prepare the dataset for training in chunks to save memory.
    """
    print(f"Loading dataset from {file_path}...")
    
    try:
        chunks = []
        for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size), desc="Loading chunks"):
            if 'input' not in chunk.columns or 'response' not in chunk.columns:
                raise ValueError(f"Dataset missing required columns. Found: {chunk.columns.tolist()}")
            chunks.append(chunk)
        df = pd.concat(chunks)
        del chunks
        optimize_memory()
        return df
    except Exception as e:
        print(f"Error loading dataset: {e}")
        exit(1)

def prepare_dataset_for_zephyr(df, tokenizer, max_length=512, batch_size=1000):
    """
    Prepare the dataset for Zephyr fine-tuning with batch processing.
    Uses Zephyr's chat template: <|user|> and <|assistant|> markers.
    """
    print("Preparing dataset for Zephyr...")
    
    df = df.dropna()
    df = df[df['input'].str.len() > 10]
    df = df[df['response'].str.len() > 10]
    
    formatted_data = []
    
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_df = df.iloc[i:i+batch_size]
        
        for _, row in batch_df.iterrows():
            conversation = f"<|user|>\n{row['input']}<|assistant|>\n{row['response']}"
            inputs = tokenizer(
                conversation,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            )
            formatted_data.append({
                "input_ids": inputs["input_ids"][0].tolist(),
                "attention_mask": inputs["attention_mask"][0].tolist(),
                "labels": inputs["input_ids"][0].tolist()
            })
        
        if (i // batch_size) % 5 == 0:
            optimize_memory()
    
    dataset = Dataset.from_list(formatted_data)
    return dataset

def setup_qlora_config():
    """
    Set up QLoRA configuration for memory-efficient fine-tuning.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

def train_zephyr_model(train_dataset, val_dataset, output_dir, base_model_path, batch_size=1, epochs=3, gradient_accumulation_steps=16):
    """Fine-tune the Zephyr-7B model using QLoRA with CPU fallback support."""
    print("Initializing Zephyr-7B model with 4-bit quantization...")
    
    # Check bitsandbytes version and configuration
    if not check_bitsandbytes_version():
        print("\nERROR: bitsandbytes not properly configured. Falling back to CPU-only mode.")
        return None, None
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    try:
        # Aggressively clean memory before loading model
        optimize_memory()
        
        # Load the locally downloaded zephyr-7b-beta model with conservative settings
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use fp16 for memory efficiency
            low_cpu_mem_usage=True      # Optimize CPU memory usage
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    except Exception as e:
        if "bitsandbytes" in str(e):
            print("\nWARNING: bitsandbytes GPU support not available. Attempting CPU-only mode...")
            try:
                # Try loading without quantization for CPU-only mode
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_path,
                    device_map="cpu",
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    max_memory={0: "8GB"},  # Limit memory usage per device
                    offload_folder="offload_folder"  # Enable disk offloading
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_path)
                print("Successfully loaded model in CPU-only mode")
            except Exception as cpu_e:
                print(f"Critical error loading model in CPU-only mode: {cpu_e}")
                return None, None
        else:
            print(f"Critical error loading model: {e}")
            return None, None
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = prepare_model_for_kbit_training(model)
    lora_config = setup_qlora_config()
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=100,
        max_grad_norm=0.5,  # Reduced for better stability
        eval_steps=200,
        save_steps=500,
        save_total_limit=2,
        evaluation_strategy="steps",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=1,
        dataloader_pin_memory=False,
        gradient_checkpointing=True,  # Enable gradient checkpointing
        optim="adamw_torch_fused"  # Use fused optimizer
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )
    
    print("Starting model training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training failed: {e}")
        return None, None
    
    optimize_memory()
    
    model_path = os.path.join(output_dir, "zephyr_7b_finetuned")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Model and tokenizer saved to {model_path}")
    
    return model, tokenizer

def convert_to_gguf(model_path, output_path):
    """
    Convert the fine-tuned model to GGUF format for use with llama.cpp.
    """
    print(f"Converting model to GGUF format at {output_path}...")
    
    try:
        cmd = [
            "python", "-m", "llama_cpp.convert_hf_to_gguf",
            model_path,
            "--outfile", output_path,
            "--outtype", "q4_0"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print(f"Successfully converted model to GGUF format at {output_path}")
            print(f"GGUF file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"Error converting model: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Conversion timed out after 1 hour.")
        return False
    except Exception as e:
        print(f"Exception during model conversion: {str(e)}")
        return False

def evaluate_model(model_path, tokenizer, test_dataset, num_samples=5):
    """
    Evaluate the fine-tuned model on a subset of the test dataset using llama.cpp.
    """
    print("\nEvaluating model on sample conversations...")
    
    try:
        from llama_cpp import Llama
        model = Llama(model_path=model_path, n_ctx=2048, n_threads=4)
    except Exception as e:
        print(f"Failed to load GGUF model: {e}")
        return
    
    test_samples = test_dataset.shuffle().select(range(min(num_samples, len(test_dataset))))
    
    for i, sample in enumerate(test_samples):
        full_text = tokenizer.decode(sample["input_ids"], skip_special_tokens=True)
        input_text = full_text.split("<|assistant|>")[0].replace("<|user|>\n", "").strip()
        
        prompt = f"<|user|>\n{input_text}<|assistant|>\n"
        response = model(prompt, max_tokens=150, temperature=0.7, top_p=0.9, top_k=40)
        
        print(f"\nSample {i+1}:")
        print(f"Input: {input_text}")
        print(f"Generated: {response['choices'][0]['text']}")
        
        optimize_memory()

if __name__ == "__main__":
    # Define file paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(model_dir), "data")
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")
    output_dir = os.path.join(model_dir, "output")
    base_model_path = os.path.join(model_dir, "output", "zephyr_7b_finetuned")  # Path to existing finetuned model
    gguf_output_path = os.path.join(model_dir, "zephyr-7b-q4.gguf")
    
    # Check bitsandbytes version before proceeding
    if not check_bitsandbytes_version():
        print("\nPlease install or upgrade bitsandbytes and run the script again.")
        exit(1)
    
    # Check if required files exist
    for path in [train_path, val_path, test_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found at {path}. Please run preprocess.py first.")
            exit(1)
    
    if not os.path.exists(base_model_path):
        print(f"Error: Base model not found at {base_model_path}.")
        print("Please ensure the Zephyr-7B model is downloaded to the correct location.")
        exit(1)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading tokenizer...")
    try:
        # Try loading from local path first
        if os.path.exists(base_model_path):
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        else:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit(1)
    
    train_df = load_dataset(train_path)
    val_df = load_dataset(val_path)
    test_df = load_dataset(test_path)
    
    train_dataset = prepare_dataset_for_zephyr(train_df, tokenizer)
    val_dataset = prepare_dataset_for_zephyr(val_df, tokenizer)
    test_dataset = prepare_dataset_for_zephyr(test_df, tokenizer)
    
    del train_df, val_df, test_df
    optimize_memory()
    
    model, tokenizer = train_zephyr_model(
        train_dataset,
        val_dataset,
        output_dir,
        base_model_path,
        batch_size=1,
        epochs=3,
        gradient_accumulation_steps=16
    )
    
    gguf_success = convert_to_gguf(
        os.path.join(output_dir, "zephyr_7b_finetuned"),
        gguf_output_path
    )
    
    if gguf_success:
        evaluate_model(gguf_output_path, tokenizer, test_dataset, num_samples=5)
    else:
        print("Skipping evaluation due to GGUF conversion failure.")
    
    print("\nModel training, conversion, and evaluation complete!")
    print("Next step: Run 'python api/app.py' to start the FastAPI server")