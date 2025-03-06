import os
import torch
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from llama_cpp import Llama
from typing import List, Optional
import faiss
import gc
import time
import json
from sentence_transformers import SentenceTransformer
from utils import genzify_response, is_mental_health_query

# Create necessary directories
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="TheraPeek API",
    description="Gen Z Mental Health Chatbot API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS with specific origins for security
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    max_age=600
)

# Global variables
model = None
tokenizer = None
emotion_classifier = None
faiss_index = None
sentence_encoder = None
conversation_store = {}
max_context_length = 5  # Maximum number of conversation turns to keep

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "message": "I'm feeling anxious about my exams",
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "user_id": "user123"
            }
        }

class ChatResponse(BaseModel):
    response: str
    emotion: str
    conversation_id: str
    timestamp: str
    message_id: str

    class Config:
        schema_extra = {
            "example": {
                "response": "I understand exam anxiety can be tough. Let's talk about it!",
                "emotion": "empathetic",
                "conversation_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2024-01-20T15:30:00Z",
                "message_id": "msg123"
            }
        }

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int

def optimize_memory():
    """Optimize memory usage for machines with limited RAM (8GB)."""
    gc.collect(2)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / 1024 / 1024
    print(f"Current memory usage: {current_memory:.2f} MB")
    
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.5)
    
    if current_memory > 3500:
        print("WARNING: High memory usage detected. Performing aggressive cleanup...")
        for i in range(3):
            gc.collect(i)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        if platform.system() == "Darwin":
            try:
                import ctypes
                lib = ctypes.CDLL("libSystem.B.dylib")
                lib.malloc_trim(0)
            except Exception:
                pass
        
        if current_memory > 5000:
            print("CRITICAL: Memory usage high. Pausing to free memory...")
            import time
            time.sleep(3)
            gc.collect(2)

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

def load_models():
    """Load Zephyr-7B model, emotion classifier, and initialize FAISS index."""
    global model, tokenizer, emotion_classifier, faiss_index, sentence_encoder
    
    try:
        # Load Zephyr-7B model using llama.cpp with optimized settings
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "model", "zephyr-7b-q4.gguf")
        model = Llama(model_path=model_path, n_ctx=2048, n_threads=4, n_batch=512)
        
        # Load emotion classifier with optimized batch size
        emotion_classifier = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=1,
            batch_size=8
        )
        
        # Initialize sentence encoder for semantic search
        sentence_encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize FAISS index with optimized parameters
        dimension = sentence_encoder.get_sentence_embedding_dimension()
        faiss_index = faiss.IndexFlatL2(dimension)
        
        print("Models loaded successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize models and components on startup."""
    load_models()

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with emotion detection and context memory."""
    try:
        # Detect emotion in user message
        emotion_result = emotion_classifier(request.message)
        detected_emotion = emotion_result[0][0]["label"]
        
        # Prepare conversation context
        if request.conversation_id and request.conversation_id in conversation_store:
            context = conversation_store[request.conversation_id]
        else:
            context = []
            request.conversation_id = str(time.time())
        
        # Encode user message for semantic search
        if faiss_index.ntotal > 0:
            message_embedding = sentence_encoder.encode([request.message])[0]
            _, similar_indices = faiss_index.search(np.array([message_embedding]), k=2)
            
            # Add relevant past context if available
            for idx in similar_indices[0]:
                if idx < len(context):
                    context.insert(0, context[idx])
        
        # Generate response using Zephyr-7B with optimized prompt
        prompt = f"Previous context: {json.dumps(context[-3:])}"
        prompt += f"\nUser emotion: {detected_emotion}"
        prompt += f"\nUser: {request.message}"
        prompt += "\nAssistant:"
        
        response = model(prompt, max_tokens=150, temperature=0.7, top_p=0.9, top_k=40)
        
        # Process response through Gen Z filter
        genz_response = genzify_response(response["choices"][0]["text"], 
                                       mental_health_context=is_mental_health_query(request.message))
        
        # Update conversation store with memory management
        new_exchange = {"user": request.message, "assistant": genz_response}
        context.append(new_exchange)
        
        # Keep only recent context to manage memory
        conversation_store[request.conversation_id] = context[-max_context_length:]
        
        # Update FAISS index with new exchange
        exchange_text = f"{new_exchange['user']} {new_exchange['assistant']}"
        exchange_embedding = sentence_encoder.encode([exchange_text])[0]
        faiss_index.add(np.array([exchange_embedding]))
        
        # Optimize memory after processing
        optimize_memory()
        
        return ChatResponse(
            response=genz_response,
            emotion=detected_emotion,
            conversation_id=request.conversation_id
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)