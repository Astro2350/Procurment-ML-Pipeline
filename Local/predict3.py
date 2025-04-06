import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

# -----------------------------
# Base Paths and Experiment Discovery
# -----------------------------
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"
results_dir = os.path.join(base_path, "results")
# Discover newest experiment folder
experiment_dirs = [os.path.join(results_dir, d) for d in os.listdir(results_dir)
                   if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("experiment_")]
if not experiment_dirs:
    raise ValueError(f"No experiment_* folders found in {results_dir}")
experiment_dirs.sort(key=os.path.getmtime)
latest_experiment_dir = experiment_dirs[-1]
checkpoint_path = os.path.join(latest_experiment_dir, "best_model.pth")
print("========== PREDICT SCRIPT START ==========")
print(f"Base path: {base_path}")
print(f"Results dir: {results_dir}")
print(f"Newest experiment directory: {latest_experiment_dir}")
print(f"Checkpoint path: {checkpoint_path}")

# -----------------------------
# Device Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -----------------------------
# Load Processed Test Data
# -----------------------------
processed_test_data_path = os.path.join(base_path, "processed_test_data.csv")
print(f"Reading processed test data from: {processed_test_data_path}")
df = pd.read_csv(processed_test_data_path)
print(f"Loaded {len(df)} rows from '{processed_test_data_path}'.")

# -----------------------------
# Load Master Label Encoder
# -----------------------------
master_encoder_path = os.path.join(base_path, "master_label_encoder.pkl")
print(f"Loading master label encoder from: {master_encoder_path}")
with open(master_encoder_path, "rb") as f:
    master_label_encoder = pickle.load(f)
num_classes = len(master_label_encoder.classes_)
print(f"Number of classes in master label encoder: {num_classes}")

# -----------------------------
# Initialize Tokenizer and DistilBERT
# -----------------------------
print("Initializing DistilBERT tokenizer and model...")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# -----------------------------
# Model Definition
# -----------------------------
class HierarchicalCategoryModel(nn.Module):
    def __init__(self, bert_model, cat_vocab_sizes, hier_vocab_sizes,
                 num_classes, dropout_rate=0.1, embedding_dim=32):
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
        self.fc1 = nn.Linear(self.total_input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.batch_norm1 = nn.BatchNorm1d(512)
        self.batch_norm2 = nn.BatchNorm1d(256)
    def forward(self, input_ids, attention_mask, cat_features, hier_features, amount):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        bert_cls = self.bert_dropout(bert_output.last_hidden_state[:, 0, :])
        cat_emb = torch.cat([emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)], dim=1)
        hier_emb = torch.cat([emb(hier_features[:, i]) for i, emb in enumerate(self.hier_embeddings)], dim=1)
        combined = torch.cat([bert_cls, cat_emb, hier_emb, amount], dim=1)
        x = self.dropout(self.relu(self.batch_norm1(self.fc1(combined))))
        x = self.dropout(self.relu(self.batch_norm2(self.fc2(x))))
        logits = self.fc3(x)
        return logits

print("Defining HierarchicalCategoryModel for prediction...")

# -----------------------------
# Construct Model
# -----------------------------
cat_vocab_sizes = [10000, 10000]
hier_vocab_sizes = [10000]*6
model = HierarchicalCategoryModel(
    bert_model=bert_model,
    cat_vocab_sizes=cat_vocab_sizes,
    hier_vocab_sizes=hier_vocab_sizes,
    num_classes=num_classes,
    dropout_rate=0.1,
    embedding_dim=32
).to(device)
print("Created HierarchicalCategoryModel with:")
print(f"  cat_vocab_sizes = {cat_vocab_sizes}")
print(f"  hier_vocab_sizes = {hier_vocab_sizes}")
print(f"  num_classes      = {num_classes}")

# -----------------------------
# Load Checkpoint (Strict Load)
# -----------------------------
print(f"Loading checkpoint from: {checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"], strict=True)
print("Checkpoint loaded successfully with strict=True.")
model.eval()
print("Model set to eval mode.")

# -----------------------------
# Batched Inference
# -----------------------------
batch_size = 32
predictions = []
print(f"Starting inference with batch_size={batch_size}...")

for start_idx in tqdm(range(0, len(df), batch_size), desc="Predicting"):
    end_idx = start_idx + batch_size
    batch_df = df.iloc[start_idx:end_idx]
    texts = batch_df["combined_text"].tolist()
    encoding = tokenizer(
        texts,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    cat_cols = ["hospital_system_id_encoded", "primary_category_id_encoded"]
    cat_feats = torch.tensor(batch_df[cat_cols].values, dtype=torch.long).to(device)
    hier_cols = [
        "category0_encoded", "category1_encoded", "category2_encoded",
        "category3_encoded", "category4_encoded", "category5_encoded"
    ]
    hier_feats = torch.tensor(batch_df[hier_cols].values, dtype=torch.long).to(device)
    if "amount" in batch_df.columns:
        amount_data = batch_df["amount"].values.reshape(-1, 1)
    else:
        amount_data = np.zeros((len(batch_df), 1), dtype=float)
    amount = torch.tensor(amount_data, dtype=torch.float).to(device)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cat_features=cat_feats,
            hier_features=hier_feats,
            amount=amount
        )
    pred_indices = torch.argmax(outputs, dim=1).cpu().numpy()
    # Use the master label encoder for inverse transform:
    batch_preds = master_label_encoder.inverse_transform(pred_indices)
    predictions.extend(batch_preds)

df["predicted_category"] = predictions
output_path = os.path.join(latest_experiment_dir, "ap_predictions.csv")
df.to_csv(output_path, index=False)
print(f"Inference complete. Predictions saved to '{output_path}'.")
print("========== PREDICT SCRIPT END ==========")
