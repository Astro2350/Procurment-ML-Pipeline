import os
import pandas as pd
import torch
import pickle
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
from tqdm import tqdm

# Define the base path
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"

# Load processed test data (which should include the encoded columns)
processed_test_data_path = os.path.join(base_path, 'processed_test_data.csv')
df = pd.read_csv(processed_test_data_path)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load saved label encoder (for target mapping)
label_encoder_path = os.path.join(base_path, 'label_encoder.pkl')
with open(label_encoder_path, 'rb') as f:
    label_encoder = pickle.load(f)

# Load primary category encoder if necessary (to create primary_category_id_encoded)
primary_encoder_path = os.path.join(base_path, 'primary_category_encoder.pkl')
with open(primary_encoder_path, 'rb') as f:
    primary_encoder = pickle.load(f)

if 'primary_category_id_encoded' not in df.columns:
    df['primary_category_id_encoded'] = primary_encoder.transform(df['primary_category_id'])

# Initialize tokenizer and DistilBERT model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Define the HierarchicalCategoryModel (matching training settings)
class HierarchicalCategoryModel(nn.Module):
    def __init__(self, bert_model, num_cat_features=3, num_hier_features=6, num_classes=345, dropout_rate=0.2, embedding_dim=32):
        super(HierarchicalCategoryModel, self).__init__()
        self.bert = bert_model
        self.bert_dropout = nn.Dropout(0.1)
        self.bert_dim = 768
        
        # Assume fixed vocab size and embedding dimensions as in training.
        self.cat_embeddings = nn.ModuleList([nn.Embedding(10000, embedding_dim) for _ in range(num_cat_features)])
        self.hier_embeddings = nn.ModuleList([nn.Embedding(10000, embedding_dim) for _ in range(num_hier_features)])
        
        self.total_input_dim = self.bert_dim + embedding_dim * (num_cat_features + num_hier_features) + 1
        
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
        
        cat_embeddings = [emb(cat_features[:, i]) for i, emb in enumerate(self.cat_embeddings)]
        cat_embeddings = torch.cat(cat_embeddings, dim=1)
        
        hier_embeddings = [emb(hier_features[:, i]) for i, emb in enumerate(self.hier_embeddings)]
        hier_embeddings = torch.cat(hier_embeddings, dim=1)
        
        combined = torch.cat([bert_cls, cat_embeddings, hier_embeddings, amount], dim=1)
        x = self.fc1(combined)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

# Determine the number of classes from the label encoder
num_classes = len(label_encoder.classes_)
model = HierarchicalCategoryModel(bert_model, num_cat_features=3, num_hier_features=6, num_classes=num_classes).to(device)

# Load the best checkpoint from training
checkpoint_path = os.path.join(base_path, "checkpoints", "best_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# -----------------------------
# Batched Inference
# -----------------------------
batch_size = 32  # Adjust as needed
predictions = []

for start_idx in tqdm(range(0, len(df), batch_size), desc="Predicting"):
    batch_df = df.iloc[start_idx : start_idx + batch_size]
    texts = batch_df['combined_text'].tolist()
    encoding = tokenizer(
        texts,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Prepare categorical features in batch
    cat_features = torch.tensor(
        batch_df[['hospital_system_id_encoded', 'primary_category_id_encoded', 'department_name_encoded']].values,
        dtype=torch.long
    ).to(device)
    
    # Prepare hierarchical features in batch
    hier_features = torch.tensor(
        batch_df[['category0_encoded', 'category1_encoded', 'category2_encoded',
                  'category3_encoded', 'category4_encoded', 'category5_encoded']].values,
        dtype=torch.long
    ).to(device)
    
    # Prepare numeric feature 'amount'
    amount = torch.tensor(batch_df['amount'].values.reshape(-1, 1), dtype=torch.float).to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            cat_features=cat_features,
            hier_features=hier_features,
            amount=amount
        )
    predicted_indices = torch.argmax(outputs, dim=1).cpu().numpy()
    batch_predictions = label_encoder.inverse_transform(predicted_indices)
    predictions.extend(batch_predictions)

df['predicted_category'] = predictions
output_path = os.path.join(base_path, 'ap_predictions.csv')
df.to_csv(output_path, index=False)
print(f"Predictions saved to {output_path}")