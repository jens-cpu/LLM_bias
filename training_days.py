import pandas as pd
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# 1. Load and preprocess the Jigsaw dataset
df = pd.read_csv("train.csv")

# Combine all 6 labels into a single binary label: 1 if toxic in any form, else 0
df['label'] = (df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1) > 0).astype(int)

# Rename the text column for HuggingFace compatibility
df = df.rename(columns={'comment_text': 'text'})

# Keep only necessary columns
df = df[['text', 'label']]

# Convert to HuggingFace dataset
dataset = Dataset.from_pandas(df)

# 2. Initialize model and tokenizer
model_name = "GroNLP/hateBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 3. Tokenize data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Split dataset
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 5. Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./hatebert_finetuned",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True
)

# 7. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 8. Train and save
trainer.train()
trainer.save_model("./hatebert_finetuned")
tokenizer.save_pretrained("./hatebert_finetuned")
