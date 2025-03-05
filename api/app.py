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
app = FastAPI(title="TheraPeek API", description="Gen Z Mental Health Chatbot API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

class ChatResponse(BaseModel):
    response: str
    emotion: str
    conversation_id: str

def optimize_memory():
    """Optimize memory usage for machines with limited RAM."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Clear old conversations periodically
    current_time = time.time()
    for conv_id in list(conversation_store.keys()):
        if current_time - float(conv_id) > 3600:  # Remove conversations older than 1 hour
            del conversation_store[conv_id]

def load_models():
    """Load Zephyr-3B model, emotion classifier, and initialize FAISS index."""
    global model, tokenizer, emotion_classifier, faiss_index, sentence_encoder
    
    try:
        # Load Zephyr-3B model using llama.cpp with optimized settings
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                "model", "zephyr-3b-q4.gguf")
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
        
        # Generate response using Zephyr-3B with optimized prompt
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