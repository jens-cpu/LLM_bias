import pandas as pd
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import (
    Dataset,
    ClassLabel
    )
from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    confusion_matrix,
    precision_recall_curve
)
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss

# 1. Daten laden und bereinigen
df = pd.read_csv("train.csv")

# Kombinierte Labels und bereinigen
df['label'] = (df[['toxic','severe_toxic','obscene','threat','insult','identity_hate']].sum(axis=1) > 0).astype(int)
df = df.rename(columns={'comment_text': 'text'})
df = df[['text', 'label']].dropna()
df = df[df['text'].str.strip().astype(bool)]  # Leere Texte entfernen

# Klassenverteilung analysieren
print("\nKlassenverteilung:")
print(df['label'].value_counts(normalize=True))
print("\nBeispiel-Texte (toxisch):")
print(df[df['label'] == 1]['text'].sample(3).values)

# 2. Modell und Tokenizer initialisieren
model_name = "./hatebert_finetuned_v3"  
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )


# 4. Datensätze vorbereiten - KORRIGIERTE VERSION
dataset = Dataset.from_pandas(df)

# Label-Spalte in ClassLabel umwandeln
dataset = dataset.cast_column("label", ClassLabel(names=["non_toxic", "toxic"]))

# Jetzt tokenisieren (nach dem Casting)
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Stratified Split durchführen
train_test = tokenized_dataset.train_test_split(
    test_size=0.2, 
    stratify_by_column="label"
)
# 5. Klassen-Gewichte berechnen
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(df['label']),
    y=df['label']
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to("cuda" if torch.cuda.is_available() else "cpu")

# 6. Custom Model mit Focal Loss
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-bce_loss)
        loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return loss.mean()

class WeightedHateBERT(torch.nn.Module):
    def __init__(self, model_name, class_weights):
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.class_weights = class_weights
        self.loss_fct = FocalLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        if labels is not None:
            loss = self.loss_fct(outputs.logits, labels)
            outputs.loss = loss
        return outputs

# 7. Training-Argumente
training_args = TrainingArguments(
    output_dir="./hatebert_finetuned_v4",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=5e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=5,
    warmup_ratio=0.1,
    gradient_accumulation_steps=1,
    fp16=torch.cuda.is_available(),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=100,
    report_to="none"
)

# 8. Metriken und Evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Threshold-Optimierung
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    precision, recall, thresholds = precision_recall_curve(labels, probs)
    optimal_idx = np.argmax(precision * recall)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds),
        "optimal_threshold": optimal_threshold
    }

# 9. Trainer initialisieren
model = WeightedHateBERT(model_name, class_weights).to(class_weights.device)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    compute_metrics=compute_metrics
)

# 10. Training durchführen
trainer.train()
trainer.save_model("./hatebert_finetuned_v4")
tokenizer.save_pretrained("./hatebert_finetuned_v4")

# 11. Evaluation visualisieren
val_pred = trainer.predict(train_test["test"])
probs = torch.softmax(torch.tensor(val_pred.predictions), dim=1)[:, 1].numpy()

# Konfusionsmatrix
cm = confusion_matrix(train_test["test"]["label"], np.argmax(val_pred.predictions, axis=1))
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Precision-Recall-Kurve
precision, recall, _ = precision_recall_curve(train_test["test"]["label"], probs)
plt.figure(figsize=(10, 7))
plt.plot(recall, precision, marker='.')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig('precision_recall_curve.png')
plt.close()

print("\nTraining abgeschlossen. Beste Metriken:")
print(trainer.state.best_metric)
