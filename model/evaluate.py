import os
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm
import json
import matplotlib.pyplot as plt
import gc

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

def load_test_data(test_path):
    """
    Load the test dataset.
    """
    print(f"Loading test data from {test_path}...")
    df = pd.read_csv(test_path)
    
    # Verify the dataset structure
    if 'input' not in df.columns or 'response' not in df.columns:
        raise ValueError(f"Dataset missing required columns. Found: {df.columns.tolist()}")
    
    return df

def create_onnx_session(model_path):
    """
    Create an ONNX Runtime session for the quantized model.
    """
    print(f"Creating ONNX Runtime session for {model_path}...")
    
    # Set up session options for optimal performance
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.intra_op_num_threads = 1  # Limit threads for memory efficiency
    
    # Create the session
    session = ort.InferenceSession(model_path, session_options)
    
    return session

def generate_response(session, tokenizer, input_text, max_length=50, temperature=0.7, top_k=50, top_p=0.9):
    """
    Generate a response for the given input text using the ONNX model.
    Improved with temperature, top-k and top-p sampling to reduce repetition and improve diversity.
    """
    # Tokenize the input text
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Run inference with the ONNX model
    ort_inputs = {session.get_inputs()[0].name: input_ids.numpy()}
    ort_outputs = session.run(None, ort_inputs)
    
    # Process the output logits
    output_ids = torch.tensor(ort_outputs[0])
    
    # Generate response tokens with improved sampling
    response_ids = []
    for _ in range(max_length):
        # Get the next token prediction
        next_token_logits = output_ids[0, -1, :]
        
        # Apply temperature scaling
        next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            indices_to_remove = torch.topk(next_token_logits, k=top_k)[0][-1]
            next_token_logits[next_token_logits < indices_to_remove] = -float('Inf')
        
        # Apply top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = -float('Inf')
        
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
        
        next_token = torch.multinomial(probs, num_samples=1).item()
        
        # Break if we reach the end of sequence token
        if next_token == tokenizer.eos_token_id:
            break
            
        response_ids.append(next_token)
        
        # Update input_ids for next iteration
        new_input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=1)
        input_ids = new_input_ids
        
        # Run inference again
        ort_inputs = {session.get_inputs()[0].name: input_ids.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        output_ids = torch.tensor(ort_outputs[0])
    
    # Decode the generated response
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

def evaluate_model(session, tokenizer, test_df, num_samples=100):
    """
    Evaluate the model on a subset of the test dataset.
    """
    print(f"Evaluating model on {num_samples} test samples...")
    
    # Limit the number of samples for evaluation
    if len(test_df) > num_samples:
        test_df = test_df.sample(num_samples, random_state=42)
    
    results = []
    
    for _, row in tqdm(test_df.iterrows(), total=len(test_df)):
        input_text = row['input']
        expected_response = row['response']
        
        # Generate response
        generated_response = generate_response(session, tokenizer, input_text)
        
        # Store results
        results.append({
            'input': input_text,
            'expected': expected_response,
            'generated': generated_response
        })
        
        # Periodically run garbage collection
        if _ % 10 == 0:
            optimize_memory()
    
    return results

def calculate_metrics(results):
    """
    Calculate evaluation metrics based on the results.
    """
    print("Calculating evaluation metrics...")
    
    # For simplicity, we'll use a basic word overlap metric
    # In a real-world scenario, you would use more sophisticated metrics like BLEU, ROUGE, etc.
    overlap_scores = []
    
    for result in results:
        expected_words = set(result['expected'].lower().split())
        generated_words = set(result['generated'].lower().split())
        
        if len(expected_words) > 0:
            overlap = len(expected_words.intersection(generated_words)) / len(expected_words)
        else:
            overlap = 0
            
        overlap_scores.append(overlap)
    
    # Calculate average overlap score
    avg_overlap = sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0
    
    # Calculate distribution of scores
    score_distribution = {
        '0-20%': sum(1 for score in overlap_scores if score < 0.2),
        '20-40%': sum(1 for score in overlap_scores if 0.2 <= score < 0.4),
        '40-60%': sum(1 for score in overlap_scores if 0.4 <= score < 0.6),
        '60-80%': sum(1 for score in overlap_scores if 0.6 <= score < 0.8),
        '80-100%': sum(1 for score in overlap_scores if score >= 0.8)
    }
    
    metrics = {
        'average_word_overlap': avg_overlap,
        'score_distribution': score_distribution
    }
    
    return metrics, overlap_scores

def plot_results(metrics, scores, output_dir):
    """
    Plot evaluation results and save the figures.
    """
    print("Plotting evaluation results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot score distribution
    plt.figure(figsize=(10, 6))
    categories = list(metrics['score_distribution'].keys())
    values = list(metrics['score_distribution'].values())
    
    plt.bar(categories, values)
    plt.xlabel('Word Overlap Score Range')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Word Overlap Scores')
    plt.savefig(os.path.join(output_dir, 'score_distribution.png'))
    
    # Plot histogram of scores
    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=20, alpha=0.7)
    plt.xlabel('Word Overlap Score')
    plt.ylabel('Frequency')
    plt.title('Histogram of Word Overlap Scores')
    plt.axvline(metrics['average_word_overlap'], color='r', linestyle='dashed', linewidth=2)
    plt.text(metrics['average_word_overlap']+0.02, plt.ylim()[1]*0.9, f'Average: {metrics["average_word_overlap"]:.2f}')
    plt.savefig(os.path.join(output_dir, 'score_histogram.png'))
    
    print(f"Plots saved to {output_dir}")

def save_results(results, metrics, output_dir):
    """
    Save evaluation results and metrics to files.
    """
    print("Saving evaluation results...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save metrics
    with open(os.path.join(output_dir, 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    # Define file paths
    model_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(model_dir), "data")
    test_path = os.path.join(data_dir, "test.csv")
    quantized_model_path = os.path.join(model_dir, "therapeek_model_quantized.onnx")
    tokenizer_path = os.path.join(model_dir, "fine_tuned_dialogpt")
    
    # If tokenizer doesn't exist in the direct path, use the one in output directory
    if not os.path.exists(os.path.join(tokenizer_path, "config.json")):
        tokenizer_path = os.path.join(model_dir, "output", "fine_tuned_dialogpt")
    output_dir = os.path.join(model_dir, "evaluation_results")
    
    # Check if required files exist
    if not os.path.exists(test_path):
        print(f"Error: Test dataset not found at {test_path}. Please run preprocess.py first.")
        exit(1)
    
    if not os.path.exists(quantized_model_path):
        print(f"Error: Quantized model not found at {quantized_model_path}. Please run quantize.py first.")
        exit(1)
    
    if not os.path.exists(tokenizer_path):
        print(f"Error: Tokenizer not found at {tokenizer_path}. Please run train.py first.")
        exit(1)
    
    # Load test data
    test_df = load_test_data(test_path)
    
    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Create ONNX session
    session = create_onnx_session(quantized_model_path)
    
    # Evaluate model
    results = evaluate_model(session, tokenizer, test_df, num_samples=50)  # Limiting to 50 samples for efficiency
    
    # Calculate metrics
    metrics, scores = calculate_metrics(results)
    
    # Plot results
    plot_results(metrics, scores, output_dir)
    
    # Save results
    save_results(results, metrics, output_dir)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Average Word Overlap Score: {metrics['average_word_overlap']:.2f}")
    print("Score Distribution:")
    for range_name, count in metrics['score_distribution'].items():
        print(f"  {range_name}: {count} samples")
    
    print("\nModel evaluation complete!")
    print("Next step: Run 'python api/app.py' to start the Flask API server")