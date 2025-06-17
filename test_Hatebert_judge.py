import pandas as pd
from pathlib import Path
from transformers import pipeline
import torch, warnings
from datasets import load_dataset

def test_with_toxicity_prompts():
    """Testet das Modell mit wissenschaftlichen Prompts"""
    warnings.filterwarnings("ignore")
    
    # 1. Datensatz korrekt laden
    dataset = load_dataset("allenai/real-toxicity-prompts", split="train")
    
    samples = []
    for item in dataset:
        prompt = item["prompt"]
        text = prompt["text"]
        toxicity = prompt["toxicity"]
        
        if toxicity < 0.2 or toxicity > 0.8:
            samples.append({
                "text": text,
                "expected_toxicity": toxicity
            })
            if len(samples) >= 20:  # 10 sichere + 10 toxische
                break
    
    # 3. Als DataFrame konvertieren
    df = pd.DataFrame({
        "id": range(len(samples)),
        "output": [x["text"] for x in samples],
        "true_toxicity": [x["expected_toxicity"] for x in samples]
    })
    
    return df

def analyze_texts(df, output_path="toxicity_results/test_prompts_hatebert_v3.csv"):
    Path("toxicity_results").mkdir(exist_ok=True)
    
    # HateBERT Pipeline
    classifier = pipeline(
        "text-classification",
        model="./hatebert_finetuned_v3",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Analyse
    results = []
    for text in df["output"].fillna("").astype(str):
        if not text.strip():
            results.append((0, 0.0, "Empty input"))
            continue
            
        res = classifier(text)[0]
        prob = res["score"] if res["label"] == "LABEL_1" else 1 - res["score"]
        toxic = int(prob > 0.5)
        results.append((toxic, prob, f"Score: {prob:.3f}"))
    
    # Ergebnisse hinzufügen
    df["hatebert_pred"] = [r[0] for r in results]
    df["hatebert_score"] = [r[1] for r in results]
    df["hatebert_explanation"] = [r[2] for r in results]
    
    # Evaluation
    df["correct"] = (df["hatebert_pred"] == (df["true_toxicity"] > 0.5).astype(int))
    accuracy = df["correct"].mean()
    
    print(f"🔍 Genauigkeit: {accuracy:.1%}")
    print(f"✅ Ergebnisse gespeichert in {output_path}")
    df.to_csv(output_path, index=False)
    print("\nBeispiel-Output:")
    print(df[["output", "true_toxicity", "hatebert_score", "correct"]].head())
    return df

if __name__ == "__main__":
    test_df = test_with_toxicity_prompts()
    analyze_texts(test_df)
