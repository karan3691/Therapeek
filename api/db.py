import os
import motor.motor_asyncio
from pymongo import MongoClient
from datetime import datetime
import json
import asyncio
import faiss
import numpy as np
from typing import Dict, List, Optional, Any

# MongoDB connection settings
MONGODB_URI = os.environ.get('MONGODB_URI', 'mongodb://localhost:27017')
DATABASE_NAME = os.environ.get('DATABASE_NAME', 'therapeek')

# Collections
CONVERSATIONS_COLLECTION = 'conversations'
USERS_COLLECTION = 'users'
FEEDBACK_COLLECTION = 'feedback'
VECTOR_DIMENSION = 384  # Dimension for sentence embeddings

# Initialize MongoDB clients and FAISS index
async_client = None
sync_client = None
faiss_index = None

def init_faiss():
    """Initialize FAISS index for semantic search."""
    global faiss_index
    try:
        # Create a new index - using L2 distance
        faiss_index = faiss.IndexFlatL2(VECTOR_DIMENSION)
        print("FAISS index initialized successfully")
        return True
    except Exception as e:
        print(f"FAISS initialization failed: {e}")
        return False

async def init_db():
    """Initialize the MongoDB connection asynchronously."""
    global async_client
    try:
        # Create async client
        async_client = motor.motor_asyncio.AsyncIOMotorClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000  # 5 second timeout for server selection
        )
        
        # Test the connection
        await async_client.admin.command('ping')
        print("MongoDB connection established successfully")
        
        # Create indexes for better query performance
        db = async_client[DATABASE_NAME]
        await db[CONVERSATIONS_COLLECTION].create_index("conversation_id")
        await db[CONVERSATIONS_COLLECTION].create_index("user_id")
        await db[CONVERSATIONS_COLLECTION].create_index("timestamp")
        
        return True
    except Exception as e:
        print(f"MongoDB connection failed: {e}")
        return False

def init_sync_db():
    """Initialize synchronous MongoDB connection for non-async contexts."""
    global sync_client
    try:
        # Create sync client
        sync_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=5000
        )
        
        # Test the connection
        sync_client.admin.command('ping')
        print("Synchronous MongoDB connection established")
        
        return True
    except Exception as e:
        print(f"Synchronous MongoDB connection failed: {e}")
        return False

async def store_conversation(conversation_id: str, user_id: Optional[str], 
                           message: str, response: str, emotion: str,
                           message_embedding: Optional[List[float]] = None) -> bool:
    """Store a conversation exchange in MongoDB and update FAISS index."""
    if not async_client:
        print("MongoDB client not initialized")
        return False
    
    try:
        db = async_client[DATABASE_NAME]
        collection = db[CONVERSATIONS_COLLECTION]
        
        # Create document
        document = {
            "conversation_id": conversation_id,
            "user_id": user_id or "anonymous",
            "timestamp": datetime.now(),
            "message": message,
            "response": response,
            "emotion": emotion,
            "embedding": message_embedding,
            "metadata": {
                "platform": os.environ.get("PLATFORM", "web"),
                "version": os.environ.get("APP_VERSION", "1.0.0")
            }
        }
        
        # Insert the document
        result = await collection.insert_one(document)
        
        # Update FAISS index if embedding is provided
        if message_embedding and faiss_index is not None:
            vector = np.array([message_embedding], dtype=np.float32)
            faiss_index.add(vector)
        
        return bool(result.inserted_id)
    
    except Exception as e:
        print(f"Error storing conversation: {e}")
        return False

async def search_similar_conversations(query_embedding: List[float], k: int = 5) -> List[Dict]:
    """Search for similar conversations using FAISS."""
    if not async_client or not faiss_index:
        print("Clients not initialized")
        return []
    
    try:
        # Convert query to numpy array
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Search in FAISS index
        distances, indices = faiss_index.search(query_vector, k)
        
        # Get corresponding conversations from MongoDB
        db = async_client[DATABASE_NAME]
        collection = db[CONVERSATIONS_COLLECTION]
        
        similar_conversations = []
        for idx in indices[0]:
            conversation = await collection.find_one(
                {"embedding": {"$exists": True}},
                skip=int(idx)
            )
            if conversation:
                similar_conversations.append({
                    "message": conversation["message"],
                    "response": conversation["response"],
                    "emotion": conversation["emotion"]
                })
        
        return similar_conversations
    
    except Exception as e:
        print(f"Error searching similar conversations: {e}")
        return []

async def get_conversation_history(conversation_id: str, limit: int = 10) -> List[Dict]:
    """Retrieve conversation history for a specific conversation ID."""
    if not async_client:
        print("MongoDB client not initialized")
        return []
    
    try:
        db = async_client[DATABASE_NAME]
        collection = db[CONVERSATIONS_COLLECTION]
        
        # Query for conversation history
        cursor = collection.find(
            {"conversation_id": conversation_id},
            {"_id": 0}  # Exclude MongoDB ID
        ).sort("timestamp", -1).limit(limit)
        
        # Convert cursor to list
        history = await cursor.to_list(length=limit)
        return history
    
    except Exception as e:
        print(f"Error retrieving conversation history: {e}")
        return []

async def store_user_feedback(conversation_id: str, user_id: Optional[str], 
                            feedback_type: str, rating: int, 
                            comments: Optional[str] = None) -> bool:
    """Store user feedback about the conversation."""
    if not async_client:
        print("MongoDB client not initialized")
        return False
    
    try:
        db = async_client[DATABASE_NAME]
        collection = db[FEEDBACK_COLLECTION]
        
        # Create feedback document
        document = {
            "conversation_id": conversation_id,
            "user_id": user_id or "anonymous",
            "timestamp": datetime.now(),
            "feedback_type": feedback_type,
            "rating": rating,
            "comments": comments
        }
        
        # Insert the document
        result = await collection.insert_one(document)
        return bool(result.inserted_id)
    
    except Exception as e:
        print(f"Error storing user feedback: {e}")
        return False

async def get_user_conversations(user_id: str, limit: int = 20) -> List[Dict]:
    """Get all conversations for a specific user."""
    if not async_client:
        print("MongoDB client not initialized")
        return []
    
    try:
        db = async_client[DATABASE_NAME]
        collection = db[CONVERSATIONS_COLLECTION]
        
        # Get unique conversation IDs for this user
        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$conversation_id"}},
            {"$sort": {"_id.timestamp": -1}},
            {"$limit": limit}
        ]
        
        cursor = collection.aggregate(pipeline)
        conversation_ids = await cursor.to_list(length=limit)
        
        # Get the most recent message from each conversation
        result = []
        for item in conversation_ids:
            conversation_id = item["_id"]
            latest_msg = await collection.find_one(
                {"conversation_id": conversation_id},
                sort=[("timestamp", -1)]
            )
            if latest_msg:
                result.append({
                    "conversation_id": conversation_id,
                    "last_message": latest_msg["message"],
                    "last_response": latest_msg["response"],
                    "timestamp": latest_msg["timestamp"]
                })
        
        return result
    
    except Exception as e:
        print(f"Error retrieving user conversations: {e}")
        return []

# Initialize both DB and FAISS on module import
init_sync_db()
init_faiss()