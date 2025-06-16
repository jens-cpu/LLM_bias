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
model_name = "Jensvollends/hatebert-finetuned"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 3. Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256  # Optional: kürzerer Kontext
    )

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 4. Split dataset
train_test = tokenized_dataset.train_test_split(test_size=0.2)
train_dataset = train_test["train"]
eval_dataset = train_test["test"]

# 5. Compute class weights
class_weights_np = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df["label"]),
    y=df["label"]
)
class_weights = torch.tensor(class_weights_np, dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# 6. Custom model with class weights
class WeightedHateBERT(torch.nn.Module):
    def __init__(self, model_name, class_weights):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    def forward(self, **inputs):
        labels = inputs.pop("labels")
        outputs = self.model(**inputs)
        logits = outputs.logits
        loss = self.loss_fn(logits, labels)
        return {"loss": loss, "logits": logits}

model = WeightedHateBERT(model_name, class_weights)

# 7. Metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions)
    }

# 8. Training arguments
training_args = TrainingArguments(
    output_dir="./hatebert_finetuned_v2",
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    warmup_steps=200,
    weight_decay=0.01,
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True
)

# 9. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# 10. Train and save model
trainer.train()
trainer.save_model("./hatebert_finetuned_v2")
tokenizer.save_pretrained("./hatebert_finetuned_v2")
