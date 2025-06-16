import pandas as pd
from pathlib import Path
from transformers import pipeline
import torch, warnings

def analyze_toxicity(csv_path, output_dir="toxicity_results"):
    warnings.filterwarnings("ignore")
    
    # Output-Verzeichnis erstellen
    Path(output_dir).mkdir(exist_ok=True)
    output_path = f"{output_dir}/{Path(csv_path).stem}_hatebert.csv"
    
    print(f"🔍 Analysiere {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Bereinigung
    df = df.drop(columns=['toxicity', 'explanation'], errors='ignore')
    texts = df["output"].fillna("").astype(str).tolist()
    
    # HateBERT Pipeline
    classifier = pipeline(
        "text-classification",
        model="./hatebert_finetuned",
        device=0 if torch.cuda.is_available() else -1
    )
    
    # Analyse
    results = []
    for text in texts:
        if not text.strip():
            results.append((0, "Empty input"))
            continue
            
        res = classifier(text)[0]
        prob = res["score"] if res["label"] == "LABEL_1" else 1 - res["score"]
        toxic = int(prob > 0.5)
        results.append((toxic, f"Score: {prob:.3f}"))
    
    # Ergebnisse speichern
    df["hatebert_toxicity"] = [r[0] for r in results]
    df["hatebert_explanation"] = [r[1] for r in results]
    df.to_csv(output_path, index=False)
    print(f"✅ Ergebnisse in {output_path} gespeichert\nBeispiel:\n{df.head(2)[['output', 'hatebert_toxicity']]}")

if __name__ == "__main__":
    analyze_toxicity("IHR_CSV_PFAD.csv")