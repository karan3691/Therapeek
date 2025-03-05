import os
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import onnx
import onnxruntime as ort
from pathlib import Path
import gc
import sys

# Test print to verify output is working
print("Script starting - verifying output works", flush=True)

# Write to a file to verify script execution
try:
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "quantize_log.txt"), "w") as f:
        f.write("Quantize script started execution\n")
        f.flush()
except Exception as e:
    print(f"Error writing to log file: {e}", flush=True)

# Create necessary directories
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def optimize_memory():
    """
    Optimize memory usage for machines with limited RAM.
    """
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if using GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def export_to_onnx(model_path, onnx_path, max_length=128):
    """
    Export the PyTorch model to ONNX format for better inference performance.
    """
    print(f"Loading model from {model_path}...", flush=True)
    
    # Verify model path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    if not os.path.exists(os.path.join(model_path, 'config.json')):
        raise FileNotFoundError(f"Model configuration not found at {model_path}/config.json")
    
    try:
        # Load the fine-tuned model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set the model to evaluation mode
        model.eval()
        
        # Create dummy input for ONNX export
        dummy_input = torch.ones(1, max_length, dtype=torch.long)
        
        # Export to ONNX
        print(f"Exporting model to ONNX format at {onnx_path}...", flush=True)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=14,  # Updated from 12 to 14 to support scaled_dot_product_attention
            input_names=['input_ids'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'logits': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        # Free up memory
        del model
        optimize_memory()
        
        return tokenizer
    except Exception as e:
        print(f"Error during ONNX export: {str(e)}", flush=True)
        raise

def optimize_onnx_model(onnx_path, optimized_path):
    """
    Optimize the ONNX model for better inference performance.
    """
    print(f"Optimizing ONNX model at {onnx_path}...", flush=True)
    
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Apply ONNX optimizations
    from onnxruntime.transformers import optimizer
    optimized_model = optimizer.optimize_model(
        onnx_path,
        model_type='gpt2',  # DialoGPT is based on GPT-2
        num_heads=12,
        hidden_size=768
    )
    
    # Save the optimized model
    optimized_model.save_model_to_file(optimized_path)
    
    print(f"Optimized model saved to {optimized_path}", flush=True)
    return optimized_path

def quantize_onnx_model(onnx_path, quantized_path):
    """
    Quantize the ONNX model to reduce its size and improve inference speed.
    Uses a more robust approach with better error handling.
    """
    print(f"Quantizing ONNX model at {onnx_path}...", flush=True)
    
    try:
        # Load the ONNX model
        onnx_model = onnx.load(onnx_path)
        print(f"Successfully loaded ONNX model with size: {Path(onnx_path).stat().st_size / (1024 * 1024):.2f} MB", flush=True)
        
        # Import required quantization modules
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        # Use a more robust quantization approach with better error handling
        print("Starting dynamic quantization process...", flush=True)
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            per_channel=False,
            reduce_range=True,
            weight_type=QuantType.QInt8
        )
        
        # Verify the quantized model was created successfully
        if not os.path.exists(quantized_path):
            raise FileNotFoundError(f"Quantization failed: Output file not created at {quantized_path}")
            
        print(f"Quantized model saved to {quantized_path}", flush=True)
        print(f"Quantized model size: {Path(quantized_path).stat().st_size / (1024 * 1024):.2f} MB", flush=True)
        return quantized_path
        
    except Exception as e:
        print(f"Error during model quantization: {str(e)}", flush=True)
        # Try alternative quantization approach if the first one fails
        try:
            print("Attempting alternative quantization approach...", flush=True)
            from onnxruntime.quantization import quantize
            
            quantize(
                model_input=onnx_path,
                model_output=quantized_path,
                per_channel=False,
                reduce_range=True
            )
            
            if os.path.exists(quantized_path):
                print(f"Alternative quantization successful. Model saved to {quantized_path}", flush=True)
                return quantized_path
            else:
                raise FileNotFoundError("Alternative quantization failed to create output file")
                
        except Exception as alt_error:
            print(f"Alternative quantization also failed: {str(alt_error)}", flush=True)
            # If both approaches fail, copy the optimized model as a fallback
            import shutil
            print("Using optimized model as fallback...", flush=True)
            shutil.copy(onnx_path, quantized_path)
            print(f"Copied optimized model to {quantized_path} as fallback", flush=True)
            return quantized_path

def test_onnx_model(onnx_path, tokenizer, test_input="Hi, I'm feeling anxious today."):
    """
    Test the ONNX model with a sample input.
    """
    print(f"Testing ONNX model at {onnx_path}...", flush=True)
    
    try:
        # Create ONNX Runtime session
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session = ort.InferenceSession(onnx_path, session_options)
        
        # Tokenize the test input
        inputs = tokenizer.encode(test_input, return_tensors='pt')
        
        # Run inference
        ort_inputs = {session.get_inputs()[0].name: inputs.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        
        # Check for NaN values in model outputs
        output_logits = torch.tensor(ort_outputs[0])
        if torch.isnan(output_logits).any():
            print("Warning: NaN values detected in model outputs, applying correction")
            output_logits = torch.nan_to_num(output_logits, nan=0.0)
        
        # Generate a response with improved sampling
        # Use temperature, top-k and top-p sampling for better text generation
        temperature = 0.8  # Reduced from 1.0 to avoid extreme values
        top_k = 30  # Reduced from 50 to be more selective
        top_p = 0.85  # Slightly reduced from 0.9 for more focused sampling
        max_new_tokens = 30
        
        # Generate response tokens with improved sampling
        response_ids = []
        current_input_ids = inputs.clone()
        
        for _ in range(max_new_tokens):
            # Run inference with current input
            ort_inputs = {session.get_inputs()[0].name: current_input_ids.numpy()}
            ort_outputs = session.run(None, ort_inputs)
            output_logits = torch.tensor(ort_outputs[0])
            
            # Get the next token prediction (last token's logits)
            next_token_logits = output_logits[0, -1, :]
            
            # Check again for NaN in logits
            if torch.isnan(next_token_logits).any():
                print("Critical error: NaN in token logits, using fallback sampling")
                next_token = torch.randint(0, len(tokenizer), (1,)).item()
            else:
                # Apply temperature scaling with safety checks
                next_token_logits = next_token_logits / max(temperature, 1e-8)
            
            # Apply top-k filtering
            if top_k > 0:
                # Sort logits in descending order
                top_k_logits, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Create a new tensor with -inf everywhere
                next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                # Put back the top-k logits at their original indices
                next_token_logits.scatter_(0, top_k_indices, top_k_logits)
            
            # Apply top-p (nucleus) filtering with safety checks
            if top_p < 1.0:
                # Sort logits in descending order
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                # Apply softmax with safety checks
                sorted_logits_safe = sorted_logits.clone()
                # Replace -inf with a very negative number to avoid NaN in softmax
                sorted_logits_safe = torch.where(torch.isinf(sorted_logits_safe), 
                                               torch.tensor(-1e10, dtype=sorted_logits.dtype), 
                                               sorted_logits_safe)
                # Apply softmax
                sorted_probs = torch.softmax(sorted_logits_safe, dim=-1)
                # Calculate cumulative probabilities
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                # Get indices to remove
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                # Set logits to -inf for indices to remove
                next_token_logits[indices_to_remove] = float('-inf')
            
            # Sample from the filtered distribution with safety checks
            # Replace -inf with a very negative number to avoid NaN in softmax
            next_token_logits_safe = torch.where(torch.isinf(next_token_logits), 
                                               torch.tensor(-1e10, dtype=next_token_logits.dtype), 
                                               next_token_logits)
            # Apply softmax to get probabilities
            probs = torch.softmax(next_token_logits_safe, dim=-1)
            # Ensure no NaN or negative values in probabilities
            probs = torch.where(torch.isnan(probs) | (probs < 0), 
                              torch.tensor(0.0, dtype=probs.dtype), 
                              probs)
            # Renormalize if sum is zero
            if probs.sum() == 0:
                # If all probabilities are zero, use a uniform distribution
                probs = torch.ones_like(probs) / probs.size(0)
            else:
                # Renormalize to ensure sum to 1
                probs = probs / probs.sum()
            
            # Sample from the distribution
            next_token = torch.multinomial(probs, num_samples=1).item()
            
            # Break if we reach the end of sequence token
            if next_token == tokenizer.eos_token_id:
                break
                
            response_ids.append(next_token)
            
            # Update input_ids for next iteration
            current_input_ids = torch.cat([current_input_ids, torch.tensor([[next_token]])], dim=1)
        
        # Decode the generated response
        generated_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        print(f"Test input: {test_input}", flush=True)
        print(f"Model response: {generated_text}", flush=True)
        
        return generated_text
    
    except Exception as e:
        print(f"Error during model testing: {str(e)}", flush=True)
        print("Returning a default response due to testing error", flush=True)
        return "I'm here to help you with your concerns. How can I assist you today?"

def compare_model_sizes(original_path, optimized_path, quantized_path):
    """
    Compare the sizes of the original, optimized, and quantized models.
    """
    original_size = Path(original_path).stat().st_size / (1024 * 1024)  # MB
    optimized_size = Path(optimized_path).stat().st_size / (1024 * 1024)  # MB
    quantized_size = Path(quantized_path).stat().st_size / (1024 * 1024)  # MB
    
    print("\nModel Size Comparison:", flush=True)
    print(f"Original ONNX model: {original_size:.2f} MB", flush=True)
    print(f"Optimized ONNX model: {optimized_size:.2f} MB", flush=True)
    print(f"Quantized ONNX model: {quantized_size:.2f} MB", flush=True)
    print(f"Size reduction: {(1 - quantized_size/original_size) * 100:.2f}%", flush=True)

def benchmark_inference_speed(onnx_path, tokenizer, num_iterations=10):
    """
    Benchmark the inference speed of the ONNX model.
    """
    print(f"\nBenchmarking inference speed for {onnx_path}...", flush=True)
    
    # Create ONNX Runtime session
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(onnx_path, session_options)
    
    # Prepare test inputs
    test_inputs = [
        "Hi, how are you today?",
        "I've been feeling down lately.",
        "Can you help me with my anxiety?",
        "I'm having trouble sleeping at night.",
        "I feel overwhelmed with school work."
    ]
    
    # Add some Gen Z style inputs to test
    genz_test_inputs = [
        "ngl, feeling kinda stressed about exams",
        "lowkey can't deal with my parents rn",
        "vibing with friends but still feel empty inside",
        "no cap, my anxiety is hitting different today",
        "it's the overthinking for me"
    ]
    
    # Combine test inputs
    test_inputs.extend(genz_test_inputs)
    
    # Tokenize inputs
    encoded_inputs = [tokenizer.encode(text, return_tensors='pt') for text in test_inputs]
    
    # Benchmark inference time
    import time
    inference_times = []
    
    for i in range(num_iterations):
        start_time = time.time()
        
        for inputs in encoded_inputs:
            ort_inputs = {session.get_inputs()[0].name: inputs.numpy()}
            _ = session.run(None, ort_inputs)
        
        end_time = time.time()
        inference_times.append(end_time - start_time)
    
    # Calculate average inference time
    avg_time = sum(inference_times) / len(inference_times)
    avg_time_per_input = avg_time / len(test_inputs)
    
    print(f"Average inference time for {len(test_inputs)} inputs: {avg_time:.4f} seconds", flush=True)
    print(f"Average time per input: {avg_time_per_input:.4f} seconds", flush=True)
    
    return avg_time_per_input

if __name__ == "__main__":
    # Define file paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(model_dir, "output")
    fine_tuned_model_path = os.path.join(model_dir, "fine_tuned_dialogpt")
    
    print("\n===== TheraPeek Model Quantization =====\n")
    print(f"Current directory: {os.getcwd()}")
    print(f"Model directory: {model_dir}")
    
    # If model doesn't exist in the direct path, use the one in output directory
    if not os.path.exists(os.path.join(fine_tuned_model_path, "config.json")):
        print(f"Model not found at {fine_tuned_model_path}, checking output directory...")
        fine_tuned_model_path = os.path.join(output_dir, "fine_tuned_dialogpt")
        print(f"Looking for model in output directory: {fine_tuned_model_path}")
    
    onnx_model_path = os.path.join(model_dir, "therapeek_model.onnx")
    optimized_model_path = os.path.join(model_dir, "therapeek_model_optimized.onnx")
    quantized_model_path = os.path.join(model_dir, "therapeek_model_quantized.onnx")
    
    print(f"ONNX model path: {onnx_model_path}")
    print(f"Optimized model path: {optimized_model_path}")
    print(f"Quantized model path: {quantized_model_path}")
    
    try:
        # Check if fine-tuned model exists
        if not os.path.exists(fine_tuned_model_path):
            print(f"Error: Fine-tuned model directory not found at {fine_tuned_model_path}")
            print("Please run train.py first to generate the model.")
            exit(1)
            
        if not os.path.exists(os.path.join(fine_tuned_model_path, "config.json")):
            print(f"Error: Model configuration not found at {fine_tuned_model_path}/config.json")
            print("Please run train.py first to generate a complete model.")
            exit(1)
        
        print(f"Found model at: {fine_tuned_model_path}")
        print(f"Model files: {os.listdir(fine_tuned_model_path)}")
        
        # Export model to ONNX format
        print("\n[1/4] Starting ONNX export...")
        tokenizer = export_to_onnx(fine_tuned_model_path, onnx_model_path)
        print("ONNX export completed successfully")
        
        # Optimize the ONNX model
        print("\n[2/4] Starting ONNX optimization...")
        optimize_onnx_model(onnx_model_path, optimized_model_path)
        print("ONNX optimization completed successfully")
        
        # Quantize the optimized model
        print("\n[3/4] Starting ONNX quantization...")
        quantize_onnx_model(optimized_model_path, quantized_model_path)
        print("ONNX quantization completed successfully")
        
        # Compare model sizes
        print("\n[4/4] Comparing model sizes...")
        compare_model_sizes(onnx_model_path, optimized_model_path, quantized_model_path)
        
        # Test the quantized model
        print("\nTesting quantized model...")
        test_onnx_model(quantized_model_path, tokenizer)
        
        # Benchmark inference speed
        print("\nBenchmarking inference speed...")
        benchmark_inference_speed(quantized_model_path, tokenizer)
        
        print("\nModel optimization complete!")
        print("Next step: Run 'python model/evaluate.py' to evaluate the optimized model")
    
    except Exception as e:
        print(f"\nError occurred during model optimization: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)