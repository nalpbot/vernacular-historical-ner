import argparse
from transformers import pipeline

def run_inference(text: str, model_path: str):
    nlp = pipeline("ner", model=model_path, aggregation_strategy="simple")
    results = nlp(text)
    print("\n--- Extracted Historical Entities ---")
    for entity in results:
        print(f"Entity: {entity['word']} | Label: {entity['entity_group']} | Score: {entity['score']:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER on historical text.")
    parser.add_argument("--input", type=str, required=True, help="Text snippet to analyze.")
    parser.add_argument("--model", type=str, default="bert-base-multilingual-cased", help="Model path or HF repo ID.")
    args = parser.parse_args()

    run_inference(args.input, args.model)