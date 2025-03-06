import os
import gc
import torch
import psutil
import platform
import argparse
import subprocess
from pathlib import Path

def optimize_memory():
    """Optimize memory usage for machines with limited RAM (8GB).
    Implements extremely aggressive memory management techniques for MacBooks.
    Specifically tuned for Zephyr-7B model conversion.
    """
    gc.collect(2)  # Full collection including oldest generation
    
    # Clear GPU/MPS cache based on device availability
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.5)  # More restrictive GPU memory limit
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
        # Apple Silicon specific optimizations
        if platform.system() == "Darwin" and platform.processor() == "arm":
            torch.mps.set_per_process_memory_fraction(0.4)  # More conservative for M1/M2
    
    # Monitor memory usage
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

def convert_to_gguf(model_path, output_path, quantization='q4_0'):
    """Convert a fine-tuned Zephyr model to GGUF format.
    
    Args:
        model_path: Path to the fine-tuned model directory
        output_path: Path for the output GGUF file
        quantization: Quantization type (q4_0, q4_1, q5_0, q5_1, q8_0)
    
    Returns:
        bool: True if conversion was successful, False otherwise
    """
    print(f"Converting model from {model_path} to GGUF format...")
    print(f"Output path: {output_path}")
    print(f"Quantization: {quantization}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Run the conversion command
    cmd = [
        "python", "-m", "llama_cpp.convert_hf_to_gguf",
        "--outfile", output_path,
        "--outtype", quantization,
        model_path
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print output for debugging
    if result.stdout:
        print("Command output:")
        print(result.stdout)
    
    if result.stderr:
        print("Command errors:")
        print(result.stderr)
    
    if result.returncode == 0:
        optimize_memory()  # Call memory optimization after successful conversion
        print(f"Successfully converted model to GGUF format at {output_path}")
        print(f"GGUF file size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")
        return True
    else:
        print(f"Failed to convert model to GGUF format. Error code: {result.returncode}")
        return False

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Convert fine-tuned Zephyr model to GGUF format')
    parser.add_argument('--model_path', type=str, default='output/zephyr_7b_finetuned',
                        help='Path to the fine-tuned model directory')
    parser.add_argument('--output_path', type=str, default='zephyr-7b-q4.gguf',
                        help='Path for the output GGUF file')
    parser.add_argument('--quantization', type=str, default='q4_0',
                        help='Quantization type (q4_0, q4_1, q5_0, q5_1, q8_0)')
    return parser.parse_args()

def main():
    """Main function to run the conversion process."""
    args = parse_arguments()
    success = convert_to_gguf(
        model_path=args.model_path,
        output_path=args.output_path,
        quantization=args.quantization
    )
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())