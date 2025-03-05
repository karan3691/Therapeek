import os
import sys
import argparse
import subprocess
import torch
import gc
from pathlib import Path

# Create necessary directories
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def optimize_memory():
    """Optimize memory usage for machines with limited RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert fine-tuned Zephyr model to GGUF format')
    parser.add_argument('--model_path', type=str, default='output/zephyr_3b_finetuned',
                        help='Path to the fine-tuned model directory')
    parser.add_argument('--output_path', type=str, default='zephyr-3b-q4.gguf',
                        help='Path for the output GGUF file')
    parser.add_argument('--quantization', type=str, default='q4_0',
                        help='Quantization type (q4_0, q4_1, q5_0, q5_1, q8_0)')
    return parser.parse_args()

def convert_to_gguf(model_path, output_path, quant_type='q4_0'):
    """Convert the fine-tuned model to GGUF format for use with llama.cpp."""
    print(f"Converting model from {model_path} to GGUF format...")
    
    # Verify model path exists
    if not os.path.exists(model_path):
        print(f"Error: Model directory not found at {model_path}")
        return False
    
    try:
        # Use llama.cpp's conversion script
        cmd = [
            "python", "-m", "llama_cpp.convert_hf_to_gguf",
            model_path,
            "--outfile", output_path,
            "--outtype", quant_type
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully converted model to GGUF format at {output_path}")
            print(f"GGUF file size: {Path(output_path).stat().st_size / (1024 * 1024):.2f} MB")
            return True
        else:
            print(f"Error converting model: {result.stderr}")
            return False
    except Exception as e:
        print(f"Exception during model conversion: {str(e)}")
        return False

def verify_gguf_model(model_path):
    """Verify the GGUF model can be loaded with llama.cpp."""
    try:
        from llama_cpp import Llama
        print(f"Verifying GGUF model at {model_path}...")
        
        # Try to load the model with minimal settings
        model = Llama(model_path=model_path, n_ctx=512, n_threads=1)
        
        # Test with a simple prompt
        test_prompt = "<|user|>\nHello, how are you?<|assistant|>\n"
        response = model(test_prompt, max_tokens=10)
        
        print(f"Model verification successful. Sample response: {response['choices'][0]['text']}")
        return True
    except Exception as e:
        print(f"Error verifying GGUF model: {str(e)}")
        return False

def main():
    """Main function to convert model to GGUF format."""
    args = parse_arguments()
    
    # Resolve paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(model_dir, args.model_path)
    output_path = os.path.join(model_dir, args.output_path)
    
    # Convert model to GGUF format
    success = convert_to_gguf(model_path, output_path, args.quantization)
    
    if success:
        # Verify the converted model
        verify_success = verify_gguf_model(output_path)
        if verify_success:
            print("\nModel conversion and verification complete!")
            print(f"The 4-bit quantized GGUF model is ready at: {output_path}")
            print("Next step: Run 'python api/app.py' to start the FastAPI server")
        else:
            print("\nModel conversion succeeded but verification failed.")
    else:
        print("\nModel conversion failed.")
    
    # Clean up memory
    optimize_memory()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())