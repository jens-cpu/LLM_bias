import json
import pandas as pd
import torch
from datasets import load_dataset
from transformers import pipeline
from detoxify import Detoxify
from tqdm import tqdm

# Device setup
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'cuda' if device == 0 else 'cpu'}")

# Load personas from JSONL
def load_personas(path, limit=None):
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]
        if limit:
            records = records[:limit]
    return pd.DataFrame(records)

print("Loading personas...")
persona_df = load_personas("persona_reduced.jsonl", limit=30)
print(f"{len(persona_df)} personas loaded.")

# Load ContextEval dataset
print("Loading ContextEval...")
context_df = load_dataset("allenai/ContextEval", split="validation").to_pandas()
context_df = context_df.sample(n=30, random_state=42)
print(f"{len(context_df)} context samples loaded.")

# Load models
generator = pipeline("text-generation", model="facebook/opt-125m", device=device)
sentiment = pipeline("sentiment-analysis", device=device)
tox_model = Detoxify("original")

# Prompt builder
def build_prompt(persona_row, context_row):
    p = persona_row
    age = p.get("age", "unknown age")
    gender = p.get("sex", "person")
    location = p.get("place of birth", "an unknown place")
    quirks = p.get("defining quirks", "")
    hobby = p.get("personal time", "")
    persona = f"This is a {age}-year-old {gender} from {location}. Known for: {quirks}. They enjoy: {hobby}."

    context = context_row["context"]
    question = context_row["question"]
    answer = context_row["answer"]

    return f"{persona}\nContext: {context}\nQ: {question}\nA: {answer}\nExplain why this answer is correct:\n"

# Generate and evaluate
results = []

for _, p_row in tqdm(persona_df.iterrows(), total=len(persona_df), desc="Processing personas"):
    for _, c_row in context_df.iterrows():
        prompt = build_prompt(p_row, c_row)
        try:
            eos_id = generator.tokenizer.eos_token_id or 50256
            out = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.8, top_p=0.9, pad_token_id=eos_id)
            gen_text = out[0]["generated_text"].split("Explain why this answer is correct:")[-1].strip()
        except Exception as e:
            gen_text = f"Generation error: {e}"

        try:
            tox = tox_model.predict(gen_text)
        except:
            tox = {"toxicity": None, "severe_toxicity": None, "identity_attack": None}

        try:
            sent = sentiment(gen_text)[0]
        except:
            sent = {"label": "ERROR", "score": None}

        results.append({
            "persona_id": p_row.get("id", ""),
            "gender": p_row.get("sex", ""),
            "religion": p_row.get("religion", ""),
            "location": p_row.get("place of birth", ""),
            "question": c_row["question"],
            "answer": c_row["answer"],
            "prompt": prompt,
            "generated_explanation": gen_text,
            "toxicity": tox.get("toxicity"),
            "severe_toxicity": tox.get("severe_toxicity"),
            "identity_attack": tox.get("identity_attack"),
            "sentiment_label": sent.get("label"),
            "sentiment_score": sent.get("score"),
        })

# Save results
df_out = pd.DataFrame(results)
df_out.to_csv("persona_contexteval_combined.csv", index=False, encoding="utf-8-sig")
print("✅ All done! Output saved to 'persona_contexteval_combined.csv'")
