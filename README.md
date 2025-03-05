# TheraPeek - Gen Z Mental Health Chatbot

This project implements a fine-tuned DialoGPT model optimized for mental health conversations with Gen Z users. The model is specifically designed to run efficiently on MacBooks with limited RAM (8GB).

## Features

- Fine-tuned DialoGPT model for mental health conversations
- Memory-optimized for MacBooks with 8GB RAM
- Flask-based RESTful API
- React frontend for real-time chat interactions
- Quantized model for efficient inference
- Gen Z conversation dataset integration for better relatability

## Model Training Pipeline

1. Download and preprocess the mental health dataset: `python data/download_dataset.py`
2. Download and preprocess the Gen Z conversation dataset: `python data/download_genz_dataset.py`
3. Preprocess and combine all datasets: `python data/preprocess.py`
4. Fine-tune the DialoGPT model: `python model/train.py`
5. Quantize the model for efficient inference: `python model/quantize.py`
6. Evaluate the model performance: `python model/evaluate.py`

## Memory Optimization

This project includes several memory optimization techniques to ensure it runs efficiently on machines with limited RAM (8GB):

- Chunked data loading and processing
- Batch processing for dataset preparation
- Gradient accumulation during training
- Aggressive garbage collection
- Memory usage monitoring
- Model quantization for inference

## API Usage

Start the Flask API server:

```bash
python api/app.py
```

The API will be available at `http://localhost:5000/api/chat`

## Frontend Development

Start the React development server:

```bash
cd frontend
npm install
npm start
```

The frontend will be available at `http://localhost:3000`

## Response Time Optimization

The model has been optimized for faster response times through:

- ONNX Runtime optimization
- Model quantization (int8)
- Efficient inference parameters
- Reduced context window size

## Requirements

See `requirements.txt` for the full list of dependencies.