# predict.py
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def load_pipeline(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True)
    parser.add_argument("--model", type=str, default="./hatebert_finetuned_v5")
    args = parser.parse_args()

    pipe = load_pipeline(args.model)
    result = pipe(args.text)[0]

    score = result['score']
    label = result['label']
    toxic_flag = 1 if label.lower() == 'toxic' or score > 0.5 else 0

    print(f"Text: {args.text}")
    print(f"Label: {label} | Score: {score:.3f} → toxic_flag={toxic_flag}")
