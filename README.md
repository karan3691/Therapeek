# TheraPeek - Gen Z Mental Health Chatbot

TheraPeek is a fine-tuned Zephyr-7B model optimized for mental health conversations with Gen Z users. The model is specifically designed to run efficiently on MacBooks with limited RAM (8GB), providing accessible mental health support with a Gen Z-friendly communication style.

> **Note:** This repository contains only the scripts and code needed to train and deploy the model. The actual trained and quantized model files are not included due to their large size, which could cause crashes on machines with limited RAM. Users can follow the instructions below to train and quantize their own model.

## Features

- Fine-tuned Zephyr-7B model for mental health conversations
- Memory-optimized for MacBooks with 8GB RAM
- Flask-based RESTful API
- React frontend for real-time chat interactions
- Quantized model (4-bit) for efficient inference
- Gen Z conversation style with context-aware responses
- Mental health-specific language patterns

## Project Structure

```
therapeek/
├── api/           # Flask API server
├── data/          # Training datasets
├── frontend/      # React web interface
├── model/         # Model training and inference scripts
└── requirements.txt
```

## Model Training Pipeline

> **Important:** The trained model files are not included in this repository. The following steps allow you to train and quantize your own model.

1. Download and preprocess the mental health dataset:
   ```bash
   python data/download_dataset.py
   ```

2. Preprocess and combine all datasets:
   ```bash
   python data/preprocess.py
   ```

3. Fine-tune the Zephyr-7B model:
   ```bash
   python model/finetune_zephyr.py
   ```
   > Note: This step requires significant computational resources. Consider using a cloud GPU service if your local machine has limited resources.

4. Quantize the model for efficient inference:
   ```bash
   python model/quantize.py
   ```

5. Evaluate the model performance:
   ```bash
   python model/evaluate.py
   ```

## Memory Optimization

The project includes several memory optimization techniques to ensure efficient operation on machines with limited RAM:

- Chunked data loading and processing
- Batch processing for dataset preparation
- Gradient accumulation during training
- Aggressive garbage collection
- Memory usage monitoring
- Model quantization (4-bit) for inference

## Response Time Optimization

The model has been optimized for faster response times through:

- ONNX Runtime optimization
- Model quantization (4-bit)
- Efficient inference parameters
- Reduced context window size

## Frontend Development

Start the React development server:

```bash
cd frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

## API Server

Start the Flask API server:

```bash
cd api
pip install -r requirements.txt
python app.py
```

The API server will be available at `http://localhost:5000`

## Model Deployment

Once you've trained and quantized your model following the steps above, it can be used with llama.cpp:

```python
from llama_cpp import Llama

# Load the model
model = Llama(model_path="zephyr-7b-q4.gguf", n_ctx=2048, n_threads=4)

# Generate a response
response = model("User: I've been feeling really anxious lately\nAssistant:", max_tokens=150)
print(response['choices'][0]['text'])
```

## Requirements

See `requirements.txt` for the full list of dependencies.

## License

This project is for research purposes only. Please use responsibly and ethically for mental health support.
