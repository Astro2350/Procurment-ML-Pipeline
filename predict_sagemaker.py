#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm

def main():
    print("========== PREDICT SCRIPT START ==========")
    
    # ------------------------------------------------------------------
    # 1. Define local container paths (from pipeline or Processing job)
    # ------------------------------------------------------------------
    input_data_path = os.path.join("/opt/ml/processing", "input", "test_data")
    input_model_path = os.path.join("/opt/ml/processing", "input", "model")
    output_path = os.path.join("/opt/ml/processing", "output")
    
    # We'll assume:
    # - /opt/ml/processing/input/test_data/processed_test_data.csv
    # - /opt/ml/processing/input/model/best_model.pth
    # - /opt/ml/processing/input/model/master_label_encoder.pkl
    
    test_csv = os.path.join(input_data_path, "processed_test_data.csv")
    checkpoint_path = os.path.join(input_model_path, "best_model.pth")
    encoder_path = os.path.join(input_model_path, "master_label_encoder.pkl")
    
    # ------------------------------------------------------------------
    # 2. Load Processed Test Data
    # ------------------------------------------------------------------
    print(f"Reading test data from: {test_csv}")
    df = pd.read_csv(test_csv)
    print(f"Loaded {len(df)} rows of test data.")
    
    # ------------------------------------------------------------------
    # 3. Load Master Label Encoder
    # ------------------------------------------------------------------
    print(f"Loading master label encoder from: {encoder_path}")
    with open(encoder_path, "rb") as f:
        master_label_encoder = pickle.load(f)
    num_classes = len(master_label_encoder.classes_)
    print(f"Number of classes in label encoder: {num_classes}")
    
    # ------------------------------------------------------------------
    # 4. Device Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # ------------------------------------------------------------------
    # 5. Initialize Tokenizer and DistilBERT
    # ------------------------------------------------------------------
    print("Initializing DistilBERT tokenizer and model backbone...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    
    # ------------------------------------------------------------------
    # 6. Model Definition (Mirroring your training code)
    # ------------------------------------------------------------------
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
    
    cat_vocab_sizes = [10000, 10000]  # e.g. hospital + primary cat
    hier_vocab_sizes = [10000] * 6    # for category0..5
    print("Building model architecture for inference...")
    model = HierarchicalCategoryModel(
        bert_model=bert_model,
        cat_vocab_sizes=cat_vocab_sizes,
        hier_vocab_sizes=hier_vocab_sizes,
        num_classes=num_classes
    ).to(device)
    
    # ------------------------------------------------------------------
    # 7. Load checkpoint from local path
    # ------------------------------------------------------------------
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    print("Checkpoint loaded successfully & model set to eval mode.")
    
    # ------------------------------------------------------------------
    # 8. Batched Inference
    # ------------------------------------------------------------------
    batch_size = 32
    predictions = []
    confidence_scores = []
    
    print(f"Starting inference (batch_size={batch_size})...")
    num_rows = len(df)
    
    # We expect columns like:
    #   'combined_text', 'hospital_system_id_encoded', 'primary_category_id_encoded',
    #   'category0_encoded', 'category1_encoded', ... 'category5_encoded',
    #   'amount' (optional)
    
    for start_idx in tqdm(range(0, num_rows, batch_size), desc="Predicting"):
        end_idx = start_idx + batch_size
        batch_df = df.iloc[start_idx:end_idx]

        # Convert text to DistilBERT input
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
        
        # Categorical features
        cat_cols = ["hospital_system_id_encoded", "primary_category_id_encoded"]
        cat_feats = torch.tensor(batch_df[cat_cols].values, dtype=torch.long).to(device)
        
        # Hierarchical features
        hier_cols = [
            "category0_encoded", "category1_encoded", "category2_encoded",
            "category3_encoded", "category4_encoded", "category5_encoded"
        ]
        hier_feats = torch.tensor(batch_df[hier_cols].values, dtype=torch.long).to(device)
        
        # 'amount' if present, else zeros
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
        
        probs = torch.softmax(outputs, dim=1)
        batch_confidences = torch.max(probs, dim=1)[0].cpu().numpy()
        confidence_scores.extend(batch_confidences.tolist())
        
        pred_indices = torch.argmax(probs, dim=1).cpu().numpy()
        batch_preds = master_label_encoder.inverse_transform(pred_indices)
        predictions.extend(batch_preds)

    df["predicted_category"] = predictions
    df["confidence_score"] = confidence_scores
    
    # Optionally separate high/low confidence
    confidence_threshold = 0.8
    low_conf_df = df[df["confidence_score"] < confidence_threshold].copy()
    high_conf_df = df[df["confidence_score"] >= confidence_threshold].copy()
    
    # ------------------------------------------------------------------
    # 9. Save Predictions
    # ------------------------------------------------------------------
    # We'll store all predictions in /opt/ml/processing/output
    # That gets uploaded to S3 by the pipeline job
    pred_all_path = os.path.join(output_path, "ap_predictions_all.csv")
    pred_low_path = os.path.join(output_path, "ap_predictions_low_confidence.csv")
    pred_high_path = os.path.join(output_path, "ap_predictions_high_confidence.csv")
    
    print(f"Saving all predictions to: {pred_all_path}")
    df.to_csv(pred_all_path, index=False)
    
    print(f"Saving low confidence to: {pred_low_path}")
    low_conf_df.to_csv(pred_low_path, index=False)
    
    print(f"Saving high confidence to: {pred_high_path}")
    high_conf_df.to_csv(pred_high_path, index=False)
    
    print("========== PREDICT SCRIPT END ==========")

if __name__ == "__main__":
    main()