import os
import torch
import onnxruntime as ort
from transformers import AutoTokenizer

MODEL_PATH = 'model/therapeek_model_quantized.onnx'
TOKENIZER_PATH = 'model/fine_tuned_dialogpt'

def validate_model():
    # Load resources
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    session = ort.InferenceSession(MODEL_PATH)

    # Test input
    test_input = "I'm feeling anxious about my final exams"
    inputs = tokenizer(test_input, return_tensors='pt').input_ids.numpy()

    # Run inference
    outputs = session.run(None, {'input_ids': inputs})
    logits = torch.tensor(outputs[0])

    # Validate outputs
    if torch.isnan(logits).any():
        print("NaN detected in model outputs! Applying correction...")
        logits = torch.nan_to_num(logits, nan=0.0)

    # Generation parameters
    gen_config = {
        'temperature': 0.7,
        'top_k': 50,
        'top_p': 0.9,
        'max_length': 100
    }

    # Generate response
    output = tokenizer.decode(
        torch.multinomial(
            torch.softmax(logits[:, -1, :] / gen_config['temperature'], dim=-1),
            1
        ).squeeze(),
        skip_special_tokens=True
    )

    print(f"Test Input: {test_input}")
    print(f"Model Output: {output}")
    print(f"Output Stats - Mean: {logits.mean():.2f}, Std: {logits.std():.2f}")

if __name__ == '__main__':
    validate_model()