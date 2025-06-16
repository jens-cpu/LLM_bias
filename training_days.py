import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight

# 1. Daten laden und vorbereiten
df = pd.read_csv("train.csv")
df['label'] = (df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1) > 0).astype(int)
df = df.rename(columns={'comment_text': 'text'})[['text', 'label']]

# 2. Modell und Tokenizer
model_name = "Jensvollends/hatebert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenisierung mit richtiger Formatierung
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"  # Wichtig für Custom Model
    )

# 4. Datensätze vorbereiten
dataset = Dataset.from_pandas(df)
tokenized_dataset = dataset.map(tokenize_function, batched=True)
train_test = tokenized_dataset.train_test_split(test_size=0.2)

# 5. Klassenweights
weights = compute_class_weight(
    "balanced",
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights = torch.tensor(weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

# 6. Optimiertes Custom Model
class WeightedHateBERT(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs

model = WeightedHateBERT(model_name).to(class_weights.device)

# 7. Training
training_args = TrainingArguments(
    output_dir="./hatebert_finetuned_v3",
    evaluation_strategy="steps",
    per_device_train_batch_size=8,  # Reduziert bei Speicherproblemen
    gradient_accumulation_steps=2,  # Simuliert größere Batchs
    fp16=torch.cuda.is_available(),  # Beschleunigt Training
    dataloader_pin_memory=False  # Behebt Hanging-Probleme
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    compute_metrics=lambda p: {
        "accuracy": accuracy_score(p.label_ids, np.argmax(p.predictions, axis=1)),
        "f1": f1_score(p.label_ids, np.argmax(p.predictions, axis=1))
    }
)

trainer.train()
trainer.save_model("./hatebert_finetuned_v3")
tokenizer.save_pretrained("./hatebert_finetuned_v3")