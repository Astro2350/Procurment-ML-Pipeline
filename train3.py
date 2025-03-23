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
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import DistilBertTokenizer, DistilBertModel

# -----------------------------
# Configuration
# -----------------------------
PLOT_CONFUSION_MATRIX = True
RANDOM_SEED = 42

def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(RANDOM_SEED)

# -----------------------------
# Base Paths
# -----------------------------
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"
results_dir = os.path.join(base_path, "results")
master_encoder_path = os.path.join(base_path, "master_label_encoder.pkl")  # persistent label encoder

# -----------------------------
# Create New Experiment Folder
# -----------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
this_experiment_dir = os.path.join(results_dir, f"experiment_{timestamp}")
os.makedirs(this_experiment_dir, exist_ok=True)
print("="*50)
print("TRAINING SCRIPT WITH INCREMENTAL LEARNING (RETAINING OLD CLASSES)")
print(f"New experiment directory: {this_experiment_dir}")
print("="*50)

# -----------------------------
# Initialize TensorBoard SummaryWriter
# -----------------------------
tb_log_dir = os.path.join(this_experiment_dir, "tensorboard_logs")
writer = SummaryWriter(log_dir=tb_log_dir)
print(f"TensorBoard logs will be saved to: {tb_log_dir}")

# -----------------------------
# Discover Previous Experiment (for resuming training)
# -----------------------------
def find_previous_experiment(base_dir, exclude_path):
    experiment_dirs = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path) and name.startswith("experiment_"):
            if os.path.normpath(full_path) != os.path.normpath(exclude_path):
                experiment_dirs.append(full_path)
    if not experiment_dirs:
        return None
    experiment_dirs.sort(key=os.path.getmtime)
    return experiment_dirs[-1]

previous_experiment_dir = find_previous_experiment(results_dir, this_experiment_dir)
if previous_experiment_dir:
    print(f"Found previous experiment directory: {previous_experiment_dir}")
    possible_checkpoint = os.path.join(previous_experiment_dir, "best_model.pth")
    if os.path.isfile(possible_checkpoint):
        print(f"Existing checkpoint found at: {possible_checkpoint}")
    else:
        print("No best_model.pth in that folder. Will train from scratch.")
        possible_checkpoint = None
else:
    print("No previous experiment folder found. Training from scratch.")
    possible_checkpoint = None

# -----------------------------
# Load Processed Training Data
# -----------------------------
processed_data_path = os.path.join(base_path, 'processed_train_data.csv')
df = pd.read_csv(processed_data_path)
print(f"Loaded training data: {processed_data_path} | Rows: {len(df)}")

# -----------------------------
# Load or Initialize Master Label Encoder
# -----------------------------
if os.path.exists(master_encoder_path):
    with open(master_encoder_path, 'rb') as f:
        master_label_encoder = pickle.load(f)
    print(f"Loaded master label encoder from: {master_encoder_path}")
else:
    # If no master encoder exists, initialize one from the current data’s target column.
    from sklearn.preprocessing import LabelEncoder
    master_label_encoder = LabelEncoder()
    master_label_encoder.fit(df['matched_category_id'])
    print("No master label encoder found. Created a new one from current data.")

# -----------------------------
# Update Master Label Encoder with New Data
# -----------------------------
# Get the unique labels in the new training data (from the original target column, not the encoded one)
new_labels = np.unique(df['matched_category_id'])
# Get the union of old and new labels
all_labels = np.unique(np.concatenate([master_label_encoder.classes_, new_labels]))
print(f"Master label set (before update): {master_label_encoder.classes_}")
print(f"New labels from data: {new_labels}")
print(f"Union of labels: {all_labels}")

# Refit the master label encoder on the union
master_label_encoder.fit(all_labels)
# Save the updated master label encoder back to disk
with open(master_encoder_path, 'wb') as f:
    pickle.dump(master_label_encoder, f)
print(f"Updated master label encoder saved to: {master_encoder_path}")

# Now, re-encode the target column using the updated master encoder
df['matched_category_id_encoded'] = master_label_encoder.transform(df['matched_category_id'])
num_classes = len(master_label_encoder.classes_)
print(f"Using fixed number of classes: {num_classes}")

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
    print("Handling outliers in 'amount' column using 'clip' method.")
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

# -----------------------------
# Construct Model with Fixed Output Dimension
# -----------------------------
cat_vocab_sizes = [10000, 10000]
hier_vocab_sizes = [10000]*6
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
model = HierarchicalCategoryModel(
    bert_model=bert_model,
    cat_vocab_sizes=cat_vocab_sizes,
    hier_vocab_sizes=hier_vocab_sizes,
    num_classes=num_classes,  # from master_label_encoder
    dropout_rate=0.1,
    embedding_dim=32
).to(device)
print(f"Created HierarchicalCategoryModel with {num_classes} output classes.")

# -----------------------------
# Function to Expand (or Shrink) Final Layer if Needed
# -----------------------------
def adjust_final_layer(model, new_num_classes):
    old_fc3 = model.fc3
    old_num_classes = old_fc3.out_features
    if new_num_classes != old_num_classes:
        new_fc3 = nn.Linear(old_fc3.in_features, new_num_classes)
        # If expanding, copy old weights
        num_to_copy = min(old_num_classes, new_num_classes)
        new_fc3.weight.data[:num_to_copy] = old_fc3.weight.data[:num_to_copy]
        new_fc3.bias.data[:num_to_copy] = old_fc3.bias.data[:num_to_copy]
        model.fc3 = new_fc3
        print(f"Adjusted final layer from {old_num_classes} to {new_num_classes} classes.")
    else:
        print("Final layer size remains unchanged.")

# Check if new master encoder has changed the number of classes compared to the current model's final layer.
current_num_classes = model.fc3.out_features
if num_classes != current_num_classes:
    adjust_final_layer(model, num_classes)
else:
    print("Final layer already has correct number of classes.")

# -----------------------------
# Compute Global Class Weights (using full label set)
# -----------------------------
global_cw = compute_class_weight(
    'balanced',
    classes=np.arange(num_classes),
    y=df['matched_category_id_encoded'].values
)
global_cw_tensor = torch.tensor(global_cw, dtype=torch.float)
print("Computed global class weights for CrossEntropyLoss.")

# -----------------------------
# Train/Validation Split
# -----------------------------
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Train set size: {len(train_df)} | Val set size: {len(val_df)}")
train_dataset = TransactionDataset(train_df, tokenizer)
val_dataset = TransactionDataset(val_df, tokenizer)
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Hyperparameters and Optimizer Setup
# -----------------------------
learning_rate = 1e-5
num_epochs = 3
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
criterion = nn.CrossEntropyLoss(weight=global_cw_tensor.to(device))
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

# -----------------------------
# Possibly Load Previous Best Model from a Known Checkpoint
# -----------------------------
if possible_checkpoint:
    print(f"Attempting to load checkpoint from previous run:\n {possible_checkpoint}")
    checkpoint = torch.load(possible_checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print("Loaded old checkpoint. Continuing from previous best.")
else:
    print("No previous checkpoint found. Training from scratch.")

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
            _, preds = torch.max(outputs, dim=1)
            all_targets.extend(targets.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(loader)
    acc = np.mean(np.array(all_preds) == np.array(all_targets))
    prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
    cm = confusion_matrix(all_targets, all_preds)
    return avg_loss, acc, f1, cm

def plot_confusion_matrix(cm, classes, filepath):
    print(f"Plotting confusion matrix with shape {cm.shape} => {filepath}")
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=False, cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()

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
    val_loss, val_acc, val_f1, cm = evaluate(model, val_loader, criterion, device, epoch, num_epochs)
    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
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
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        early_stopping_counter = 0
        best_model_state = model.state_dict()
        best_model_path = os.path.join(this_experiment_dir, "best_model.pth")
        torch.save({
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict()
        }, best_model_path)
        print(f"[Epoch {epoch+1}] Best F1 so far = {best_val_f1:.4f}. Saved best_model.pth to {best_model_path}.")
        if PLOT_CONFUSION_MATRIX and num_classes <= 100:
            inv_classes = [master_label_encoder.inverse_transform([i])[0] for i in range(num_classes)]
            cm_path = os.path.join(this_experiment_dir, f"confusion_matrix_epoch{epoch+1}.png")
            plot_confusion_matrix(cm, inv_classes, cm_path)
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