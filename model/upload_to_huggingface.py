import os
import sys
import torch
import shutil
import argparse
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder
from transformers import AutoTokenizer, AutoModelForCausalLM
import gc

# Create necessary directories
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def optimize_memory():
    """Optimize memory usage for machines with limited RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Upload TheraPeek model to Hugging Face Hub')
    parser.add_argument('--model_path', type=str, default='../model/zephyr-3b-q4.gguf',
                        help='Path to the quantized GGUF model')
    parser.add_argument('--repo_name', type=str, default='therapeek-zephyr-3b-q4',
                        help='Name for the Hugging Face repository')
    parser.add_argument('--hf_token', type=str, required=True,
                        help='Hugging Face API token')
    parser.add_argument('--username', type=str, required=True,
                        help='Hugging Face username')
    parser.add_argument('--private', action='store_true',
                        help='Make the repository private')
    return parser.parse_args()

def prepare_model_for_upload(model_path, temp_dir):
    """Prepare the model files for upload to Hugging Face."""
    print(f"Preparing model from {model_path} for upload...")
    
    # Create temporary directory for model files
    os.makedirs(temp_dir, exist_ok=True)
    
    # Copy the GGUF model file
    if os.path.exists(model_path):
        model_filename = os.path.basename(model_path)
        shutil.copy(model_path, os.path.join(temp_dir, model_filename))
        print(f"Copied {model_filename} to temporary directory")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Create model card (README.md)
    model_card = f"""
# TheraPeek - Gen Z Mental Health Chatbot

## Model Description
This is a fine-tuned and quantized version of Zephyr-3B optimized for mental health conversations with Gen Z users. 
The model is specifically designed to run efficiently on MacBooks with limited RAM (8GB).

## Features
- Fine-tuned for mental health support conversations
- Optimized for Gen Z language and slang
- 4-bit quantized for efficient inference on consumer hardware
- Compatible with llama.cpp for local deployment

## Usage

```python
from llama_cpp import Llama

# Load the model
model = Llama(model_path="zephyr-3b-q4.gguf", n_ctx=2048, n_threads=4)

# Generate a response
response = model("User: I've been feeling really anxious lately\nAssistant:", max_tokens=150)
print(response['choices'][0]['text'])
```

## Training
This model was fine-tuned on mental health conversation datasets using PEFT (QLoRA) for memory-efficient training.

## License
This model is for research purposes only. Please use responsibly and ethically for mental health support.
"""
    
    with open(os.path.join(temp_dir, "README.md"), "w") as f:
        f.write(model_card)
    
    # Create config.json with model metadata
    config = {
        "model_type": "llama",
        "architectures": ["LlamaForCausalLM"],
        "quantization_config": {
            "bits": 4,
            "group_size": 128,
            "desc": "4-bit quantized GGUF format for llama.cpp"
        },
        "model_name": "therapeek-zephyr-3b-q4",
        "language": ["en"],
        "license": "mit",
        "tags": ["mental-health", "gen-z", "chatbot", "llama.cpp", "quantized"]
    }
    
    with open(os.path.join(temp_dir, "config.json"), "w") as f:
        import json
        json.dump(config, f, indent=2)
    
    print("Model prepared for upload with README and config")
    return temp_dir

def upload_to_huggingface(prepared_dir, repo_id, token, private=False):
    """Upload the model to Hugging Face Hub."""
    print(f"Uploading model to Hugging Face Hub: {repo_id}")
    
    try:
        # Initialize Hugging Face API
        api = HfApi(token=token)
        
        # Create or get repository
        create_repo(repo_id=repo_id, token=token, private=private, exist_ok=True)
        
        # Upload the model files
        upload_folder(
            folder_path=prepared_dir,
            repo_id=repo_id,
            token=token,
            ignore_patterns=["*.pyc", "__pycache__", ".git*"],
        )
        
        print(f"Model successfully uploaded to https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"Error uploading to Hugging Face: {str(e)}")
        return False

def main():
    """Main function to upload the model to Hugging Face Hub."""
    args = parse_arguments()
    
    # Validate model path
    model_path = os.path.abspath(args.model_path)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return 1
    
    # Create temporary directory for model preparation
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_hf_upload")
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    try:
        # Prepare model for upload
        prepared_dir = prepare_model_for_upload(model_path, temp_dir)
        
        # Create repository ID
        repo_id = f"{args.username}/{args.repo_name}"
        
        # Upload to Hugging Face
        success = upload_to_huggingface(
            prepared_dir=prepared_dir,
            repo_id=repo_id,
            token=args.hf_token,
            private=args.private
        )
        
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")
        
        return 0 if success else 1
    
    except Exception as e:
        print(f"Error: {str(e)}")
        # Clean up temporary directory in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        return 1
    finally:
        # Optimize memory
        optimize_memory()

if __name__ == "__main__":
    sys.exit(main())