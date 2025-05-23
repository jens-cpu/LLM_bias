import json
import pandas as pd
import torch
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

# Load ContextEval dataset manually from local .jsonl file (only first 20 lines)
try:
    with open("all_autorater_judgements.jsonl", "r", encoding="utf-8") as f:
        lines = [next(f) for _ in range(20)]
    context_df = pd.read_json("".join(lines), lines=True)
    
    # Sample size adjustment (up to 30, but max lines read is 20 here)
    sample_size = min(30, len(context_df))
    context_df = context_df.sample(n=sample_size, random_state=42)
    
    print(f"{len(context_df)} context samples loaded (from first 20 lines).")
except Exception as e:
    print("❌ Failed to load limited ContextEval JSONL lines:")
    print(e)
    exit(1)

# Load models
generator = pipeline("text-generation", model="facebook/opt-125m", device=device)
sentiment = pipeline("sentiment-analysis", device=device)
tox_model = Detoxify("original")

# Build prompt function
def build_prompt(persona_row, context_row):
    age     = persona_row.get("age", "unknown age")
    gender  = persona_row.get("sex", "person")
    loc     = persona_row.get("place of birth", "an unknown place")
    quirks  = persona_row.get("defining quirks", "")
    hobby   = persona_row.get("personal time", "")
    
    persona = (
        f"This is a {age}-year-old {gender} from {loc}. "
        f"Known for: {quirks}. They enjoy: {hobby}."
    )

    context = context_row.get("context", "")
    query   = context_row.get("query", "")
    resp_a  = context_row.get("candidate_one_response", "")
    resp_b  = context_row.get("candidate_two_response", "")

    return (
        f"{persona}\n\n"
        f"Context:\n{context}\n\n"
        f"User asks: {query}\n\n"
        f"Response A: {resp_a}\n\n"
        f"Response B: {resp_b}\n\n"
        f"Which response would this person prefer and why?"
    )

# Prepare to store results
results = []

# Iterate over personas and context samples
for _, p_row in tqdm(persona_df.iterrows(), total=len(persona_df), desc="Processing personas"):
    for _, c_row in context_df.iterrows():
        prompt = build_prompt(p_row, c_row)
        
        try:
            eos_id = generator.tokenizer.eos_token_id or 50256
            out = generator(
                prompt,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=eos_id
            )
            gen_text = out[0]["generated_text"].split("Explain why this answer is correct:")[-1].strip()
        except Exception as e:
            gen_text = f"Generation error: {e}"

        try:
            tox = tox_model.predict(gen_text)
        except Exception:
            tox = {"toxicity": None, "severe_toxicity": None, "identity_attack": None}

        try:
            sent = sentiment(gen_text)[0]
        except Exception:
            sent = {"label": "ERROR", "score": None}

        # Append results inside the loop — p_row and c_row are defined here
        results.append({
            "persona_id": p_row.get("id", ""),
            "gender": p_row.get("sex", ""),
            "religion": p_row.get("religion", ""),
            "location": p_row.get("place of birth", ""),
            "query": c_row.get("query", ""),
            "resp_a": c_row.get("candidate_one_response", ""),
            "resp_b": c_row.get("candidate_two_response", ""),
            "chosen": c_row.get("rand_choice", None),          # 1 or 2
            "gold_eval": c_row.get("eval_judgement", None),   # human or auto judgement
            "prompt": prompt,
            "generated_explanation": gen_text,
            "toxicity": tox.get("toxicity"),
            "severe_toxicity": tox.get("severe_toxicity"),
            "identity_attack": tox.get("identity_attack"),
            "sentiment_label": sent.get("label"),
            "sentiment_score": sent.get("score"),
        })

# Save results to CSV
df_out = pd.DataFrame(results)
df_out.to_csv("persona_contexteval_combined.csv", index=False, encoding="utf-8-sig")

print("✅ All done! Output saved to 'persona_contexteval_combined.csv'")
