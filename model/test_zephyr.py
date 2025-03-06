import os
import sys
from llama_cpp import Llama
import time

# Path to the GGUF model
MODEL_PATH = 'model/zephyr-7b-q4.gguf'

def validate_zephyr_model():
    """Test the Zephyr-7B model with sample mental health queries."""
    print("\nValidating Zephyr-7B model...")
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run 'python model/finetune_zephyr.py' or 'python model/convert_to_gguf.py' first.")
        return False
    
    try:
        # Load the model with optimized settings for 8GB RAM MacBooks
        print(f"Loading model from {MODEL_PATH}...")
        start_time = time.time()
        model = Llama(
            model_path=MODEL_PATH,
            n_ctx=2048,       # Context window size
            n_threads=4,      # Number of CPU threads to use
            n_batch=512,      # Batch size for prompt processing
            use_mlock=True    # Lock memory to prevent swapping
        )
        load_time = time.time() - start_time
        print(f"Model loaded in {load_time:.2f} seconds")
        
        # Test queries focused on mental health for Gen Z
        test_queries = [
            "I'm feeling really anxious about my exams next week",
            "My friends are all hanging out without me and I feel left out",
            "I can't stop doom scrolling on social media and it's affecting my mood",
            "I feel like I'm not good enough compared to everyone on Instagram",
            "My parents don't understand my career choices and it's stressing me out"
        ]
        
        # Test each query
        for i, query in enumerate(test_queries):
            print(f"\n--- Test Query {i+1} ---")
            print(f"Input: {query}")
            
            # Format prompt according to Zephyr's expected format
            prompt = f"<|user|>\n{query}<|assistant|>\n"
            
            # Generate response with timing
            start_time = time.time()
            response = model(
                prompt,
                max_tokens=150,
                temperature=0.7,
                top_p=0.9,
                top_k=40,
                stop=["<|user|>", "<|system|>"]
            )
            gen_time = time.time() - start_time
            
            # Print results
            print(f"Response: {response['choices'][0]['text']}")
            print(f"Generation time: {gen_time:.2f} seconds")
            print(f"Tokens generated: {len(response['choices'][0]['text'].split())}")
            
            # Small delay between queries to allow memory cleanup
            time.sleep(1)
        
        print("\nModel validation complete!")
        return True
        
    except Exception as e:
        print(f"Error during model validation: {str(e)}")
        return False

def test_memory_usage():
    """Test memory usage during model loading and inference."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        
        # Memory before loading
        mem_before = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage before loading model: {mem_before:.2f} MB")
        
        # Load model
        model = Llama(model_path=MODEL_PATH, n_ctx=2048, n_threads=4)
        
        # Memory after loading
        mem_after_load = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage after loading model: {mem_after_load:.2f} MB")
        print(f"Memory increase: {mem_after_load - mem_before:.2f} MB")
        
        # Test inference
        prompt = "<|user|>\nHow can I manage anxiety?<|assistant|>\n"
        model(prompt, max_tokens=100)
        
        # Memory after inference
        mem_after_inference = process.memory_info().rss / 1024 / 1024
        print(f"Memory usage after inference: {mem_after_inference:.2f} MB")
        print(f"Additional memory for inference: {mem_after_inference - mem_after_load:.2f} MB")
        
        return True
    except ImportError:
        print("psutil not installed, skipping memory usage test")
        return False
    except Exception as e:
        print(f"Error during memory usage test: {str(e)}")
        return False

if __name__ == '__main__':
    print("Zephyr-7B Model Test Utility")
    print("===========================")
    
    # Validate the model
    model_valid = validate_zephyr_model()
    
    if model_valid and '--memory-test' in sys.argv:
        # Run memory usage test if requested
        print("\nRunning memory usage test...")
        test_memory_usage()
    
    print("\nTest complete!")
    if model_valid:
        print("The Zephyr-7B model is working correctly and ready for use with the API.")
        print("Next step: Run 'python api/app.py' to start the FastAPI server")
    else:
        print("There were issues with the Zephyr-7B model. Please check the errors above.")