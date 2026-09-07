import json
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi

def format_ner_dataset(raw_filepath: str) -> DatasetDict:
    """
    Transforms raw annotated JSON into Hugging Face Dataset format.
    Expects JSON structure: [{"tokens": ["Word1", "Word2"], "ner_tags": [0, 1]}]
    """
    with open(raw_filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Convert to Hugging Face Dataset
    hf_dataset = Dataset.from_list(data)
    
    # Create train/val split
    split_dataset = hf_dataset.train_test_split(test_size=0.2, seed=42)
    
    return DatasetDict({
        "train": split_dataset["train"],
        "validation": split_dataset["test"]
    })

def push_to_hub(dataset_dict: DatasetDict, repo_id: str):
    """Pushes the processed dataset to Hugging Face."""
    dataset_dict.push_to_hub(repo_id, private=False)
    print(f"Dataset successfully uploaded to https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    # Example usage
    # ds = format_ner_dataset("data/processed/annotated_texts.json")
    # push_to_hub(ds, "your-username/south-asian-historical-ner")
    pass