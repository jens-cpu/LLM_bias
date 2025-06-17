#!/usr/bin/env python
"""
predict.py  –  schnelle Inferenz für HateBERT fine‑tuned v4

Beispiele
---------
# Einzelnen Text vorhersagen
python predict.py --text "You are totally awful!"

# CSV-Datei vorhersagen (Spalte 'text'), Ergebnis wird output.csv
python predict.py --csv input.csv --out output.csv
"""
import argparse, sys, csv, pathlib
import pandas as pd
import torch
from transformers import pipeline

# --------------------------------------------------
def get_classifier(model_dir="./hatebert_finetuned_v4"):
    """Lädt Pipeline aus v4-Ordner (GPU, wenn verfügbar)."""
    return pipeline(
        task="text-classification",
        model=model_dir,
        tokenizer=model_dir,
        device=0 if torch.cuda.is_available() else -1,
        truncation=True,
        max_length=512,
        batch_size=16,
        return_all_scores=True   # gibt beide Klassen-Scores aus
    )

# --------------------------------------------------
def predict_text(text, clf, threshold=0.5):
    """Gibt Flag + Score zurück."""
    res = clf(text)[0]                  # [{'label': 'LABEL_0', ...},{'label':'LABEL_1', ...}]
    score = next(r["score"] for r in res if r["label"] == "LABEL_1")
    flag  = int(score > threshold)
    return flag, score

# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--text", help="Einzelner Text")
    g.add_argument("--csv",  help="CSV mit Spalte 'text'")
    parser.add_argument("--out", help="Ausgabedatei (optional)")
    parser.add_argument("--model", default="./hatebert_finetuned_v4",
                        help="Pfad zum gespeicherten Modellordner")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Entscheidungsschwellwert für 'toxisch'")
    args = parser.parse_args()

    clf = get_classifier(args.model)

    # 1) Nur ein einzelner Text
    if args.text:
        flag, score = predict_text(args.text, clf, args.threshold)
        print(f"Text: {args.text}\nScore: {score:.3f}  → toxic_flag={flag}")
        sys.exit(0)

    # 2) CSV-Datei
    df = pd.read_csv(args.csv)
    if "text" not in df.columns:
        sys.exit("⚠️  Die CSV muss eine Spalte 'text' enthalten.")

    preds, scores = [], []
    for txt in df["text"].fillna(""):
        flag, sc = predict_text(txt, clf, args.threshold)
        preds.append(flag); scores.append(sc)

    df["toxicity_score"] = scores
    df["toxicity_flag"]  = preds

    out_path = args.out or pathlib.Path(args.csv).with_suffix(".preds.csv")
    df.to_csv(out_path, index=False)
    print(f"✅ Ergebnisse gespeichert nach {out_path}")

# --------------------------------------------------
if __name__ == "__main__":
    main()
