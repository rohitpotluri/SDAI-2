"""Bert.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1LCnTfZibvgk6Zp46VtZutG3uMLfsuL-8
"""

import os
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

os.environ["WANDB_DISABLED"] = "true"

data = pd.read_csv("Bert_set_final_fixed.csv")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'], data['label'], test_size=0.2, random_state=42
)

# Resetting the index of train_labels and val_labels
train_labels = train_labels.reset_index(drop=True)
val_labels = val_labels.reset_index(drop=True)

tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
max_len = 100
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding=True, max_length=max_len)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding=True, max_length=max_len)

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)  # Ensure label is long type
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
val_dataset = SentimentDataset(val_encodings, val_labels)

model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    eval_strategy='steps',  # Updated from evaluation_strategy
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer  # Ensure tokenizer handling matches new guidelines if warnings arise
)

trainer.train()

import torch

model_path = "TinyBERT_model.pt"
torch.save(model, model_path)

from google.colab import files
files.download(model_path)

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertForSequenceClassification

# Load the test data
test_data = pd.read_csv("test_set.csv")

# Preprocess the test data
tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")

test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# Tokenize the test data
test_encodings = tokenizer(test_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")

# Load the saved model
model = torch.load("TinyBERT_model.pt")
model.eval()

# Predict on the test set
with torch.no_grad():
    outputs = model(**test_encodings)
    probabilities = torch.softmax(outputs.logits, dim=1)
    predictions = torch.argmax(probabilities, axis=1)

# Convert to CPU for metrics computation
predictions = predictions.cpu().numpy()
test_labels = torch.tensor(test_labels).cpu().numpy()

# Calculate Metrics
accuracy = accuracy_score(test_labels, predictions)
precision = precision_score(test_labels, predictions)
recall = recall_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions)
conf_matrix = confusion_matrix(test_labels, predictions)
roc_auc = roc_auc_score(test_labels, probabilities[:, 1])

# Print classification report
print("Classification Report:\n", classification_report(test_labels, predictions))

# Plot Confusion Matrix
def plot_confusion_matrix(cm, labels=["Negative", "Positive"]):
    fig, ax = plt.subplots()
    cax = ax.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

plot_confusion_matrix(conf_matrix)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(test_labels, probabilities[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Print Metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")
print(f"ROC-AUC Score: {roc_auc:.2f}")

# Print Confusion Matrix components
tn, fp, fn, tp = conf_matrix.ravel()
print(f"True Positives: {tp}, True Negatives: {tn}, False Positives: {fp}, False Negatives: {fn}")