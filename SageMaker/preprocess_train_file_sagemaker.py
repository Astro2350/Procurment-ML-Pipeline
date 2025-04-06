import pandas as pd
import numpy as np
import re
import nltk
import os
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# SageMaker paths
input_path = "/opt/ml/processing/input"
output_path = "/opt/ml/processing/output"

# Load input files
train_df = pd.read_csv(os.path.join(input_path, "TRAIN.csv"))
categories_df = pd.read_csv(os.path.join(input_path, "list_of_categories.csv"))

print("Preprocessing training data...")

train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')
categories_df.columns = categories_df.columns.str.strip().str.lower().str.replace(' ', '_')

merged_df = train_df.merge(categories_df, left_on='primary_category_id', right_on='id', how='left')

# Fill missing values
merged_df['name'] = merged_df['name'].fillna('')
merged_df['gl_description'] = merged_df['gl_description'].fillna('')
merged_df['memo'] = merged_df['memo'].fillna('')
merged_df['department_name'] = merged_df['department_name'].fillna('Unknown')

# Combined and preprocess text
merged_df['combined_text'] = (
    merged_df['name'] + ' ' +
    merged_df['gl_description'] + ' ' +
    merged_df['memo'] + ' ' +
    merged_df['department_name']
)
merged_df['combined_text'] = merged_df['combined_text'].apply(preprocess_text)

# Encoders
target_encoder = LabelEncoder()
merged_df['matched_category_id_encoded'] = target_encoder.fit_transform(merged_df['matched_category_id'])

primary_encoder = LabelEncoder()
merged_df['primary_category_id_encoded'] = primary_encoder.fit_transform(merged_df['primary_category_id'])

# to keep numbers consistent without dropping leading zero etc.
merged_df['hospital_system_id'] = merged_df['hospital_system_id'].astype(str)
hospital_encoder = LabelEncoder()
merged_df['hospital_system_id_encoded'] = hospital_encoder.fit_transform(merged_df['hospital_system_id'])

department_encoder = LabelEncoder()
merged_df['department_name_encoded'] = department_encoder.fit_transform(merged_df['department_name'])

category_encoders = {}
for i in range(6):
    col = f'category{i}'
    merged_df[col] = merged_df[col].fillna('Unknown')
    encoder = LabelEncoder()
    merged_df[f'{col}_encoded'] = encoder.fit_transform(merged_df[col])
    category_encoders[col] = encoder

# Remove rare classes
class_counts = merged_df['matched_category_id_encoded'].value_counts()
rare_classes = class_counts[class_counts == 1].index
filtered_df = merged_df[~merged_df['matched_category_id_encoded'].isin(rare_classes)].copy()

target_encoder = LabelEncoder()
filtered_df['matched_category_id_encoded'] = target_encoder.fit_transform(filtered_df['matched_category_id'])

# Save processed CSV
filtered_df.to_csv(os.path.join(output_path, "processed_train_data.csv"), index=False)

# Save encoders
with open(os.path.join(output_path, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(target_encoder, f)
with open(os.path.join(output_path, 'primary_category_encoder.pkl'), 'wb') as f:
    pickle.dump(primary_encoder, f)
with open(os.path.join(output_path, 'hospital_system_encoder.pkl'), 'wb') as f:
    pickle.dump(hospital_encoder, f)
with open(os.path.join(output_path, 'department_encoder.pkl'), 'wb') as f:
    pickle.dump(department_encoder, f)
for i in range(6):
    with open(os.path.join(output_path, f'category{i}_encoder.pkl'), 'wb') as f:
        pickle.dump(category_encoders[f'category{i}'], f)

print("Preprocessing complete.")
