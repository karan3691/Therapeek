import os
import pandas as pd
from datasets import load_dataset

# Define the data directory
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(data_dir, exist_ok=True)

def check_file_exists(file_name):
    """Check if a dataset file already exists."""
    return os.path.exists(os.path.join(data_dir, file_name))

def download_huggingface_dataset(dataset_id, file_name):
    """Downloads datasets from Hugging Face and saves as CSV."""
    print(f"📥 Downloading {dataset_id} dataset from Hugging Face...")
    try:
        dataset = load_dataset(dataset_id)
        df = pd.DataFrame(dataset['train'])
        dataset_path = os.path.join(data_dir, file_name)
        df.to_csv(dataset_path, index=False)
        print(f"✅ Saved: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"❌ Error downloading {dataset_id}: {e}")
        return None

if __name__ == "__main__":
    print("\n🚀 Starting dataset download process...\n")

    # Hugging Face datasets
    huggingface_datasets = [
        ("tcabanski/mental_health_counseling_conversations_rated", "mental_health_counseling_rated.csv"),
        ("Amod/mental_health_counseling_conversations", "mental_health_counseling_conversations.csv"),
        ("ZahrizhalAli/mental_health_conversational_dataset", "mental_health_conversational.csv"),
        ("jkhedri/psychology-dataset", "psychology_dataset.csv")  # NEW DATASET
    ]

    for dataset_id, file_name in huggingface_datasets:
        if not check_file_exists(file_name):
            download_huggingface_dataset(dataset_id, file_name)

    print("\n✅ All datasets downloaded successfully! You can now run `python data/preprocess.py` to process the data.\n")
