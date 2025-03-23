import os
import csv
import json
import random
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# -----------------------------
# Base Paths and Experiment Directory
# -----------------------------
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"
results_dir = os.path.join(base_path, "results")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

# -----------------------------
# Load Processed Data and Encoders
# -----------------------------
processed_data_path = os.path.join(base_path, 'processed_train_data.csv')
df = pd.read_csv(processed_data_path)

with open(os.path.join(base_path, 'label_encoder.pkl'), 'rb') as f:
    target_encoder = pickle.load(f)

num_classes = df['matched_category_id_encoded'].nunique()
print(f"Number of classes from processed data: {num_classes}")

# -----------------------------
# Use Fixed Vocabulary Sizes (to match checkpoint weights)
# -----------------------------
# In previous training, embedding layers were initialized with a fixed vocabulary size of 10000.
cat_vocab_sizes = [10000, 10000, 15000]  # For hospital_system_id, primary_category_id, department_name
hier_vocab_sizes = [10000, 10000, 10000, 10000, 10000, 10000]  # For category0 to category5

# -----------------------------
# Outlier Handling for 'amount'
# -----------------------------
def handle_outliers(df, column, method='clip'):
    if method == 'clip':
        q1, q3 = df[column].quantile(0.25), df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return df[column].clip(lower_bound, upper_bound)
    elif method == 'log':
        return np.log1p(df[column].abs())
    else:
        return df[column]

if 'amount' in df.columns:
    df['amount'] = handle_outliers(df, 'amount', method='clip')

# -----------------------------
# Initialize Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# -----------------------------
# Helper: Get a Fresh BERT Model
# -----------------------------
def get_fresh_bert():
    return DistilBertModel.from_pretrained('distilbert-base-uncased')

# -----------------------------
# Custom Dataset Class
# -----------------------------
class TransactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        encoding = self.tokenizer(
            row['combined_text'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        cat_features = torch.tensor([
            row['hospital_system_id_encoded'],
            row['primary_category_id_encoded'],
            row['department_name_encoded']
        ], dtype=torch.long)

        hier_features = torch.tensor([
            row['category0_encoded'], row['category1_encoded'], row['category2_encoded'],
            row['category3_encoded'], row['category4_encoded'], row['category5_encoded']
        ], dtype=torch.long)

        amount = torch.tensor([row['amount']] if 'amount' in row and not pd.isna(row['amount']) else [0.0], dtype=torch.float)
        target = torch.tensor(row['matched_category_id_encoded'], dtype=torch.long)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'cat_features': cat_features,
            'hier_features': hier_features,
            'amount': amount,
            'target': target
        }

# -----------------------------
# Model Architecture
# -----------------------------
class HierarchicalCategoryModel(nn.Module):
    def __init__(self, bert_model, cat_vocab_sizes, hier_vocab_sizes, num_classes,
                 dropout_rate=0.1, embedding_dim=32, hidden_dim1=512, hidden_dim2=256):
        super(HierarchicalCategoryModel, self).__init__()
        self.bert = bert_model
        self.bert_dropout = nn.Dropout(0.1)
        self.bert_dim = 768

        self.cat_embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in cat_vocab_sizes])
        self.hier_embeddings = nn.ModuleList([nn.Embedding(vocab_size, embedding_dim) for vocab_size in hier_vocab_sizes])

        cat_emb_dim = embedding_dim * len(cat_vocab_sizes)
        hier_emb_dim = embedding_dim * len(hier_vocab_sizes)

        self.total_input_dim = self.bert_dim + cat_emb_dim + hier_emb_dim + 1
        self.fc1 = nn.Linear(self.total_input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim1)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim2)

    def forward(self, input_ids, attention_mask, cat_features, hier_features, amount):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = self.bert_dropout(bert_out.last_hidden_state[:, 0, :])
        cat_emb = torch.cat([emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        hier_emb = torch.cat([emb(hier_features[:, i]) for i, emb in enumerate(self.hier_embeddings)], dim=1)
        combined = torch.cat([bert_cls, cat_emb, hier_emb, amount], dim=1)
        x = self.dropout(self.relu(self.batch_norm1(self.fc1(combined))))
        x = self.dropout(self.relu(self.batch_norm2(self.fc2(x))))
        logits = self.fc3(x)
        return logits

# -----------------------------
# Training & Evaluation Functions
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_targets, all_preds = [], []
    for batch in tqdm(loader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        cat_features = batch['cat_features'].to(device)
        hier_features = batch['hier_features'].to(device)
        amount = batch['amount'].to(device)
        targets = batch['target'].to(device)

        outputs = model(input_ids, attention_mask, cat_features, hier_features, amount)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        all_targets.extend(targets.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return avg_loss, acc, precision, recall, f1

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_targets, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_features = batch['cat_features'].to(device)
            hier_features = batch['hier_features'].to(device)
            amount = batch['amount'].to(device)
            targets = batch['target'].to(device)

            outputs = model(input_ids, attention_mask, cat_features, hier_features, amount)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    conf_matrix = confusion_matrix(all_targets, all_preds)
    return avg_loss, acc, precision, recall, f1, all_targets, all_preds, conf_matrix

def plot_confusion_matrix(cm, classes, filepath):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

# -----------------------------
# Global Class Weights
# -----------------------------
global_cw = compute_class_weight('balanced', classes=np.arange(num_classes), y=df['matched_category_id_encoded'].values)
global_cw_tensor = torch.tensor(global_cw, dtype=torch.float)

# -----------------------------
# Train/Validation Split
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_dataset = TransactionDataset(train_df, tokenizer)
val_dataset = TransactionDataset(val_df, tokenizer)

batch_size = 16  # Adjust as needed
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Hyperparameters and Model Initialization
# -----------------------------
learning_rate = 1e-5
dropout_rate = 0.1
embedding_dim = 32
num_epochs = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HierarchicalCategoryModel(
    get_fresh_bert(),
    cat_vocab_sizes=cat_vocab_sizes,
    hier_vocab_sizes=hier_vocab_sizes,
    num_classes=num_classes,
    dropout_rate=dropout_rate,
    embedding_dim=embedding_dim
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=global_cw_tensor.to(device))

# -----------------------------
# Load Existing Checkpoint for Fine-Tuning (if available)
# -----------------------------
checkpoint_path = os.path.join(base_path, "checkpoints", "best_model.pth")
if os.path.exists(checkpoint_path):
    print("Loading existing checkpoint for fine-tuning...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    
    # Remove keys with mismatched shapes:
    # - Remove fc3 keys because the final layer's shape has changed (345 -> 1125 classes)
    # - Remove cat_embeddings.2 keys because the department name embedding changed (10000 -> 15000 vocab)
    for key in list(state_dict.keys()):
        if key.startswith("fc3.") or key.startswith("cat_embeddings.2."):
            del state_dict[key]
    
    # Load the filtered state dictionary into the model
    model.load_state_dict(state_dict, strict=False)
    
    # Reinitialize the optimizer to avoid conflicts with optimizer states for the removed parameters.
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
else:
    print("No checkpoint found. Training from scratch.")
# -----------------------------
# Training Loop
# -----------------------------
best_f1 = 0
best_model_state = None
log_filepath = os.path.join(experiment_dir, "training_log.csv")
with open(log_filepath, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1'])

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    train_loss, train_acc, train_prec, train_rec, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, val_prec, val_rec, val_f1, val_targets, val_preds, cm = evaluate(model, val_loader, criterion, device)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
    
    with open(log_filepath, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])
    
    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = model.state_dict()
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }, os.path.join(experiment_dir, "best_model.pth"))
        plot_confusion_matrix(cm, [target_encoder.inverse_transform([i])[0] for i in range(num_classes)], 
                              os.path.join(experiment_dir, "confusion_matrix.png"))

print("Training complete!")
