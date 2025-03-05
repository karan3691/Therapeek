import os
import pandas as pd
import requests
import json
import time
import numpy as np
from tqdm import tqdm
import gc
from datasets import load_dataset
import kaggle
import re

# Create data directory if it doesn't exist
os.makedirs(os.path.dirname(os.path.abspath(__file__)), exist_ok=True)

def download_mental_health_dataset():
    """Downloads and prepares mental health conversation datasets from Hugging Face"""
    print("Downloading Mental Health Chat dataset from Hugging Face...")
    
    try:
        # Load the dataset from Hugging Face
        dataset = load_dataset("mpingale/mental-health-chat-dataset")
        
        # Convert to pandas DataFrame
        df = pd.DataFrame(dataset['train'])
        
        # Extract only the conversation text from HTML content
        def extract_text(html_content):
            if not isinstance(html_content, str):
                return ""
            # Remove HTML tags and extract text
            text = re.sub(r'<[^>]+>', '', html_content)
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return text
        
        # Clean the question and answer text
        df['question'] = df['questionText'].apply(extract_text)
        df['answer'] = df['answerText'].apply(extract_text)
        
        # Select only the cleaned columns
        df = df[['question', 'answer']]
        
        # Save the processed dataset
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mental_health_conversational.csv')
        df.to_csv(dataset_path, index=False)
        
        print(f"Dataset downloaded successfully and saved to: {dataset_path}")
        print(f"Sample conversation:\n{df.iloc[0]}")
        
        return [dataset_path]
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return None

def process_text_conversation_file(file_path):
    """Process text file containing Human 1/Human 2 conversations"""
    print(f"Processing text conversation file: {file_path}...")
    
    try:
        conversations = []
        current_pair = {'input': '', 'response': ''}
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing conversations"):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Human 1:'):
                if current_pair['input'] and current_pair['response']:
                    conversations.append(current_pair)
                    current_pair = {'input': '', 'response': ''}
                current_pair['input'] = line.replace('Human 1:', '').strip()
            elif line.startswith('Human 2:'):
                current_pair['response'] = line.replace('Human 2:', '').strip()
        
        # Add the last pair if complete
        if current_pair['input'] and current_pair['response']:
            conversations.append(current_pair)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(conversations)
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_chat_processed.csv')
        df.to_csv(output_path, index=False)
        
        print(f"Text conversations processed and saved to: {output_path}")
        print(f"Sample conversation:\n{df.iloc[0]}")
        
        return [output_path]
        
    except Exception as e:
        print(f"Error processing text file: {e}")
        return None

def download_human_conversation_dataset():
    """Downloads and prepares human conversation dataset from Kaggle"""
    print("Downloading Human Conversation dataset from Kaggle...")
    
    try:
        # Download dataset from Kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files('projjal1/human-conversation-training-data', path=os.path.dirname(os.path.abspath(__file__)), unzip=True)
        
        # Load and process the dataset
        dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_conversations.csv')
        df = pd.read_csv(dataset_path)
        
        # Create conversation pairs from human1 and human2
        conversations = []
        for _, row in df.iterrows():
            if pd.notna(row['human1']) and pd.notna(row['human2']):
                conversations.append({
                    'input': row['human1'],
                    'response': row['human2']
                })
        
        # Convert to DataFrame
        processed_df = pd.DataFrame(conversations)
        
        # Save the processed dataset
        processed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_conversations_processed.csv')
        processed_df.to_csv(processed_path, index=False)
        
        print(f"Dataset downloaded and processed successfully. Saved to: {processed_path}")
        print(f"Sample conversation:\n{processed_df.iloc[0]}")
        
        return [processed_path]
        
    except Exception as e:
        print(f"Error downloading Kaggle dataset: {e}")
        print("Note: To use Kaggle API, please ensure you have set up your Kaggle API credentials.")
        return None

def create_synthetic_dataset(size=1000, dataset_type='mental_health'):
    """Creates synthetic conversation datasets"""
    print(f"Creating synthetic {dataset_type} dataset...")
    
    if dataset_type == 'mental_health':
        questions = [
            "I've been feeling really down lately and can't seem to enjoy things I used to love. What should I do?",
            "How do I know if I have anxiety?",
            "I'm having trouble sleeping at night. Any tips?",
            "I feel overwhelmed with work and study. How can I manage stress better?",
            "Sometimes I feel like no one understands me. How can I communicate better?",
            "I've been having panic attacks recently. What can help?",
            "How do I support a friend who might be depressed?",
            "I'm constantly worried about the future. How can I stop overthinking?",
            "What are some good self-care practices for mental health?",
            "I feel lonely even when I'm with people. Is this normal?"
        ]
        
        responses = [
            "I'm sorry you're feeling down. This could be a sign of depression. Try to reconnect with activities you used to enjoy, even if you don't feel like it at first. Consider talking to a mental health professional who can provide proper guidance.",
            "Anxiety often manifests as persistent worry, restlessness, difficulty concentrating, and physical symptoms like rapid heartbeat or tension. If these symptoms interfere with your daily life, consider consulting a healthcare provider for an assessment.",
            "Establish a regular sleep schedule, create a relaxing bedtime routine, limit screen time before bed, avoid caffeine and alcohol in the evening, and make sure your sleeping environment is comfortable and quiet.",
            "Break large tasks into smaller, manageable steps. Practice time management techniques like the Pomodoro method. Make sure to schedule breaks and self-care. Don't hesitate to ask for help when needed.",
            "Try using 'I' statements to express your feelings without blaming others. Practice active listening. Consider therapy to develop better communication skills and understand your own needs better.",
            "During a panic attack, focus on your breathing - try the 4-7-8 technique. Ground yourself by naming things you can see, hear, and touch. Regular exercise, adequate sleep, and limiting caffeine can help prevent attacks.",
            "Listen without judgment, encourage them to seek professional help, offer specific support rather than vague offers, educate yourself about depression, and take care of your own mental health too.",
            "Practice mindfulness to stay in the present moment. Challenge negative thoughts with evidence. Set aside specific 'worry time' to contain anxious thoughts. Regular exercise and adequate sleep can also help reduce overthinking.",
            "Regular physical activity, adequate sleep, healthy eating, limiting social media, spending time in nature, practicing mindfulness, connecting with others, and engaging in activities you enjoy are all important for mental wellbeing.",
            "Feeling lonely in a crowd can happen to many people. It might indicate you're craving deeper connections rather than casual interactions. Try opening up more in conversations and seeking out people with similar interests or values."
        ]
        
        column_names = {'input': 'question', 'output': 'answer'}
        output_file = 'mental_health_synthetic.csv'
        
    else:  # Gen Z dataset
        questions = [
            "ngl, this class is lowkey stressing me out fr",
            "bruh moment when your crush leaves you on read",
            "vibing with my bestie but feeling kinda off",
            "no cap, my parents just don't get me",
            "this anxiety is hitting different today",
            "feeling sus about my future ngl",
            "can't even with school rn",
            "big yikes, just bombed that test",
            "lowkey feeling like a failure",
            "it's the overthinking for me"
        ]
        
        responses = [
            "that's a mood. maybe take some time to chill and reset?",
            "fr fr, that's rough. but don't let it live in your head rent-free",
            "it be like that sometimes. wanna talk about what's throwing your vibe off?",
            "parents are from a different era, they just don't understand the pressure we're under",
            "anxiety is mad annoying. have you tried those breathing techniques? they're not cap",
            "the future is scary af but you're not alone in feeling that way",
            "school is draining the life out of all of us. take a mental health day if you need it",
            "one test doesn't define you. you got this on the next one",
            "we're all just figuring it out. don't be too hard on yourself",
            "overthinking is my toxic trait too. try to stay in the present"
        ]
        
        column_names = {'input': 'input', 'output': 'response'}
        output_file = 'genz_conversations_processed.csv'
    
    # Create synthetic data with memory efficiency
    synthetic_data = []
    for _ in tqdm(range(size), desc="Creating synthetic data"):
        q_idx = np.random.randint(0, len(questions))
        r_idx = np.random.randint(0, len(responses))
        
        synthetic_data.append({
            column_names['input']: questions[q_idx],
            column_names['output']: responses[r_idx]
        })
        
        # Add variations for Gen Z dataset
        if dataset_type == 'genz' and len(synthetic_data) % 10 == 0:
            modifiers = ["tbh", "lowkey", "highkey", "fr", "no cap", "literally", "deadass"]
            if np.random.random() > 0.5:
                q = f"{modifiers[np.random.randint(0, len(modifiers))]} {questions[np.random.randint(0, len(questions))]}"
                synthetic_data.append({
                    column_names['input']: q,
                    column_names['output']: responses[np.random.randint(0, len(responses))]
                })
    
    # Save synthetic dataset
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_file)
    df = pd.DataFrame(synthetic_data)
    df.to_csv(dataset_path, index=False)
    
    print(f"Synthetic dataset created with {len(df)} conversations")
    print(f"Sample conversation:\n{df.iloc[0]}")
    
    return dataset_path

if __name__ == "__main__":
    print("Starting dataset download process...")
    
    # Download mental health datasets
    mental_health_datasets = download_mental_health_dataset()
    
    # Process text conversation file
    text_conversation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'human_chat.txt')
    text_conversation_datasets = None
    if os.path.exists(text_conversation_path):
        text_conversation_datasets = process_text_conversation_file(text_conversation_path)
    
    # Download human conversation dataset
    human_conversation_datasets = download_human_conversation_dataset()
    
    # Check results and create synthetic dataset if needed
    if not mental_health_datasets:
        print("\nWarning: Failed to download mental health datasets. Creating synthetic dataset.")
        synthetic_dataset = create_synthetic_dataset(size=2000, dataset_type='mental_health')
        if synthetic_dataset:
            mental_health_datasets = [synthetic_dataset]
    
    # Combine all available datasets
    all_datasets = []
    if mental_health_datasets:
        all_datasets.extend(mental_health_datasets)
    if text_conversation_datasets:
        all_datasets.extend(text_conversation_datasets)
    if human_conversation_datasets:
        all_datasets.extend(human_conversation_datasets)
    
    if all_datasets:
        print("\nAll datasets prepared successfully!")
        print(f"Available datasets: {all_datasets}")
        print("\nNext step: Run 'python data/preprocess.py' to prepare the data for training")
    else:
        print("\nError: Failed to prepare datasets")