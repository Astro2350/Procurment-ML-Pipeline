import os
import csv
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from sklearn.metrics import precision_recall_fscore_support, classification_report
import matplotlib.pyplot as plt
from transformers import DistilBertTokenizer, DistilBertModel

# -----------------------------
# Configuration
# -----------------------------
# 42 because its the answer to everything in the universe lmao
RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(RANDOM_SEED)

# -----------------------------
# Base Paths (Using S3 URI)
# -----------------------------
base_path = "s3://lxeml/CH_Test/"
results_dir = os.path.join(base_path, "results")
master_encoder_path = os.path.join(base_path, "master_label_encoder.pkl")

# -----------------------------
# Create New Experiment Folder
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
this_experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
os.makedirs(this_experiment_dir, exist_ok=True)
print("="*50)
print("TRAINING SCRIPT - Starting Fresh Training")
print(f"New experiment directory: {this_experiment_dir}")
print("="*50)

# -----------------------------
# Initialize TensorBoard SummaryWriter
# -----------------------------
tb_log_dir = os.path.join(this_experiment_dir, "tensorboard_logs")
writer = SummaryWriter(log_dir=tb_log_dir)
print(f"TensorBoard logs will be saved to: {tb_log_dir}")

# -----------------------------
# Load Processed Training Data from S3
# -----------------------------
processed_data_path = os.path.join(base_path, 'processed_train_data.csv')
df = pd.read_csv(processed_data_path)
print(f"Loaded training data from '{processed_data_path}' | Rows: {len(df)}")

# -----------------------------
# Load Master Label Encoder
# -----------------------------
with open(master_encoder_path, 'rb') as f:
    master_label_encoder = pickle.load(f)
print(f"Loaded master label encoder from: {master_encoder_path}")
num_classes = len(master_label_encoder.classes_)
print(f"Using {num_classes} classes.")

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
    print("Handling outliers in 'amount' column using clipping.")
    df['amount'] = handle_outliers(df, 'amount', method='clip')

# -----------------------------
# Initialize Tokenizer
# -----------------------------
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Initialized DistilBERT tokenizer.")

# -----------------------------
# Custom Dataset Definition
# -----------------------------
class TransactionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.data = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
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
            row['primary_category_id_encoded']
        ], dtype=torch.long)
        hier_features = torch.tensor([
            row['category0_encoded'], row['category1_encoded'], row['category2_encoded'],
            row['category3_encoded'], row['category4_encoded'], row['category5_encoded']
        ], dtype=torch.long)
        val_amount = row['amount'] if ('amount' in row and not pd.isna(row['amount'])) else 0.0
        amount = torch.tensor([val_amount], dtype=torch.float)
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
# Model Architecture Definition
# -----------------------------
class HierarchicalCategoryModel(nn.Module):
    def __init__(self, bert_model, cat_vocab_sizes, hier_vocab_sizes,
                 num_classes, dropout_rate=0.1, embedding_dim=32,
                 hidden_dim1=512, hidden_dim2=256):
        super(HierarchicalCategoryModel, self).__init__()
        self.bert = bert_model
        self.bert_dropout = nn.Dropout(0.1)
        self.bert_dim = 768
        self.cat_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in cat_vocab_sizes
        ])
        self.hier_embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, embedding_dim)
            for vocab_size in hier_vocab_sizes
        ])
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
        cat_emb = torch.cat([
            emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)
        ], dim=1)
        hier_emb = torch.cat([
            emb(hier_features[:, i]) for i, emb in enumerate(self.hier_embeddings)
        ], dim=1)
        combined = torch.cat([bert_cls, cat_emb, hier_emb, amount], dim=1)
        x = self.dropout(self.relu(self.batch_norm1(self.fc1(combined))))
        x = self.dropout(self.relu(self.batch_norm2(self.fc2(x))))
        logits = self.fc3(x)
        return logits

print("Building model...")

cat_vocab_sizes = [10000, 10000]
hier_vocab_sizes = [10000] * 6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = HierarchicalCategoryModel(
    bert_model=bert_model,
    cat_vocab_sizes=cat_vocab_sizes,
    hier_vocab_sizes=hier_vocab_sizes,
    num_classes=num_classes,
    dropout_rate=0.1,
    embedding_dim=32
).to(device)
print(f"Created HierarchicalCategoryModel with {num_classes} output classes.")

# -----------------------------
# Compute Global Class Weights
# -----------------------------
present_classes = np.unique(df['matched_category_id_encoded'].values)
weights_present = compute_class_weight('balanced', classes=present_classes, y=df['matched_category_id_encoded'].values)
global_cw = np.ones(num_classes)
for i, cls in enumerate(present_classes):
    global_cw[cls] = weights_present[i]
global_cw_tensor = torch.tensor(global_cw, dtype=torch.float)
print("Computed global class weights.")

# -----------------------------
# Train/Validation Split and Oversampling
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train set size: {len(train_df)} | Val set size: {len(val_df)}")

train_dataset = TransactionDataset(train_df, tokenizer)
val_dataset = TransactionDataset(val_df, tokenizer)

batch_size = 16
# Compute sample weights for oversampling
sample_weights = compute_sample_weight('balanced', train_df['matched_category_id_encoded'])
sample_weights = torch.tensor(sample_weights, dtype=torch.float)
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Hyperparameters and Optimizer Setup
# -----------------------------
learning_rate = 1e-5
num_epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=global_cw_tensor.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

print("No previous checkpoint loaded. Training from scratch.")

# -----------------------------
# Additional Metric Function: Top-N Accuracy
# -----------------------------
def top_n_accuracy(y_true, y_probs, n=3):
    top_n_preds = np.argsort(y_probs, axis=1)[:, -n:]
    correct = 0
    for i, true_label in enumerate(y_true):
        if true_label in top_n_preds[i]:
            correct += 1
    return correct / len(y_true)

# -----------------------------
# Training and Evaluation Functions
# -----------------------------
def train_epoch(model, loader, optimizer, criterion, device, epoch_idx, total_epochs):
    model.train()
    total_loss = 0
    all_targets, all_preds = [], []
    for batch in tqdm(loader, desc=f"Epoch {epoch_idx+1} - Training", leave=False):
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
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return avg_loss, acc, f1

def evaluate(model, loader, criterion, device, epoch_idx, total_epochs):
    model.eval()
    total_loss = 0
    all_targets, all_preds = [], []
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Epoch {epoch_idx+1} - Validation", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            cat_features = batch['cat_features'].to(device)
            hier_features = batch['hier_features'].to(device)
            amount = batch['amount'].to(device)
            targets = batch['target'].to(device)
            outputs = model(input_ids, attention_mask, cat_features, hier_features, amount)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1)
            all_probs.extend(probs.cpu().numpy())
            _, preds = torch.max(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    return avg_loss, acc, f1, np.array(all_targets), np.array(all_preds), np.array(all_probs)

# -----------------------------
# Training Loop
# -----------------------------
print("Starting training loop...")
best_val_f1 = 0
early_stopping_counter = 0
log_filepath = os.path.join(this_experiment_dir, "training_log.csv")
with open(log_filepath, 'w', newline='') as f:
    writer_csv = csv.writer(f)
    writer_csv.writerow(['Epoch', 'Train Loss', 'Train Acc', 'Train F1', 'Val Loss', 'Val Acc', 'Val F1'])

for epoch in range(num_epochs):
    print(f"\n=== EPOCH {epoch+1}/{num_epochs} ===")
    train_loss, train_acc, train_f1 = train_epoch(model, train_loader, optimizer, criterion, device, epoch, num_epochs)
    val_loss, val_acc, val_f1, y_true, y_pred, y_probs = evaluate(model, val_loader, criterion, device, epoch, num_epochs)
    
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
    
    # Log metrics to TensorBoard
    current_lr = scheduler.get_last_lr()[0]
    writer.add_scalar("LearningRate", current_lr, epoch+1)
    writer.add_scalar("Loss/Train", train_loss, epoch+1)
    writer.add_scalar("Loss/Val", val_loss, epoch+1)
    writer.add_scalar("Accuracy/Train", train_acc, epoch+1)
    writer.add_scalar("Accuracy/Val", val_acc, epoch+1)
    writer.add_scalar("F1/Train", train_f1, epoch+1)
    writer.add_scalar("F1/Val", val_f1, epoch+1)
    
    with open(log_filepath, 'a', newline='') as f:
        writer_csv = csv.writer(f)
        writer_csv.writerow([epoch+1, train_loss, train_acc, train_f1, val_loss, val_acc, val_f1])
    
    class_report = classification_report(y_true, y_pred)
    print("Classification Report:")
    print(class_report)
    report_path = os.path.join(this_experiment_dir, f"classification_report_epoch{epoch+1}.txt")
    with open(report_path, 'w') as f:
         f.write(class_report)
    
    top3_acc = top_n_accuracy(y_true, y_probs, n=3)
    top5_acc = top_n_accuracy(y_true, y_probs, n=5)
    print(f"Top-3 Accuracy: {top3_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}")
    writer.add_scalar("Top-3 Accuracy/Val", top3_acc, epoch+1)
    writer.add_scalar("Top-5 Accuracy/Val", top5_acc, epoch+1)
    
    confidences = np.max(y_probs, axis=1)
    plt.figure(figsize=(8,6))
    plt.hist(confidences, bins=50, color='blue', alpha=0.7)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title("Confidence Distribution on Validation Set")
    conf_plot_path = os.path.join(this_experiment_dir, f"confidence_distribution_epoch{epoch+1}.png")
    plt.savefig(conf_plot_path)
    plt.close()
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stopping_counter = 0
        best_model_state = model.state_dict()
        best_model_path = os.path.join(this_experiment_dir, "best_model.pth")
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }, best_model_path)
        print(f"[Epoch {epoch+1}] Best F1 = {best_val_f1:.4f}. Saved best_model.pth to {best_model_path}.")
    else:
        early_stopping_counter += 1
        print(f"No improvement in F1 for {early_stopping_counter} epoch(s).")
        if early_stopping_counter >= 3:
            print("Early stopping triggered.")
            break
    scheduler.step(val_loss)

writer.close()
print(f"Training complete! Logs saved to: {log_filepath}")
print(f"Final model and artifacts stored in: {this_experiment_dir}")