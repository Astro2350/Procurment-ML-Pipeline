import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
import os
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text: lowercases, removes non-alphabet characters,
# removes stopwords and applies lemmatization.
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Define the base path for your project
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"

# Load raw datasets: training data and categories info
categories_df = pd.read_csv(os.path.join(base_path, 'list_of_categories.csv'))
train_df = pd.read_csv(os.path.join(base_path, 'TRAIN.csv'))

print("Preprocessing training data...")

# Clean column names: remove extra spaces, convert to lowercase, and replace spaces with underscores
train_df.columns = train_df.columns.str.strip().str.lower().str.replace(' ', '_')
categories_df.columns = categories_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Merge training data with category info based on primary_category_id and categories' id
merged_df = train_df.merge(categories_df, left_on='primary_category_id', right_on='id', how='left')

# Fill missing values for text columns and create a combined text field
merged_df['name'] = merged_df['name'].fillna('')
merged_df['gl_description'] = merged_df['gl_description'].fillna('')
merged_df['memo'] = merged_df['memo'].fillna('')
merged_df['combined_text'] = merged_df['name'] + ' ' + merged_df['gl_description'] + ' ' + merged_df['memo']

# Preprocess the combined text column
merged_df['combined_text'] = merged_df['combined_text'].apply(preprocess_text)

# ---- Encode the Target Variable (matched_category_id) ----
# This is what we want the ML model to predict.
target_encoder = LabelEncoder()
merged_df['matched_category_id_encoded'] = target_encoder.fit_transform(merged_df['matched_category_id'])
print(f"Number of target classes before filtering: {len(target_encoder.classes_)}")

# ---- Encode the Primary Category for Feature Use ----
# We'll encode the primary_category_id so that it can be used as a categorical feature.
primary_encoder = LabelEncoder()
merged_df['primary_category_id_encoded'] = primary_encoder.fit_transform(merged_df['primary_category_id'])

# ---- Encode Other Feature Columns ----
# Encode hospital_system_id
merged_df['hospital_system_id'] = merged_df['hospital_system_id'].astype(str)
hospital_encoder = LabelEncoder()
merged_df['hospital_system_id_encoded'] = hospital_encoder.fit_transform(merged_df['hospital_system_id'])

# Encode department_name
merged_df['department_name'] = merged_df['department_name'].fillna('Unknown')
department_encoder = LabelEncoder()
merged_df['department_name_encoded'] = department_encoder.fit_transform(merged_df['department_name'])

# Encode hierarchical features (category0 to category5)
category_encoders = {}
for i in range(6):
    col_name = f'category{i}'
    merged_df[col_name] = merged_df[col_name].fillna('Unknown')
    encoder = LabelEncoder()
    merged_df[f'{col_name}_encoded'] = encoder.fit_transform(merged_df[col_name])
    category_encoders[col_name] = encoder

# ---- Remove Rare Classes ----
# Remove target classes (matched_category_id_encoded) that occur only once.
class_counts = merged_df['matched_category_id_encoded'].value_counts()
rare_classes = class_counts[class_counts == 1].index
filtered_df = merged_df[~merged_df['matched_category_id_encoded'].isin(rare_classes)].copy()

# Re-fit the target encoder on the filtered data so that labels become contiguous.
target_encoder = LabelEncoder()
filtered_df['matched_category_id_encoded'] = target_encoder.fit_transform(filtered_df['matched_category_id'])
num_classes = filtered_df['matched_category_id_encoded'].nunique()
print(f"Number of target classes after filtering: {num_classes}")

# ---- Save Processed Training Data and Encoders ----
processed_data_path = os.path.join(base_path, 'processed_train_data.csv')
filtered_df.to_csv(processed_data_path, index=False)

# Save the target label encoder (for matched_category_id)
with open(os.path.join(base_path, 'label_encoder.pkl'), 'wb') as f:
    pickle.dump(target_encoder, f)

# Save the primary category encoder (for feature use)
with open(os.path.join(base_path, 'primary_category_encoder.pkl'), 'wb') as f:
    pickle.dump(primary_encoder, f)

# Save other feature encoders
with open(os.path.join(base_path, 'hospital_system_encoder.pkl'), 'wb') as f:
    pickle.dump(hospital_encoder, f)

with open(os.path.join(base_path, 'department_encoder.pkl'), 'wb') as f:
    pickle.dump(department_encoder, f)

for i in range(6):
    col_name = f'category{i}'
    with open(os.path.join(base_path, f'{col_name}_encoder.pkl'), 'wb') as f:
        pickle.dump(category_encoders[col_name], f)

print(f"Training data preprocessing complete and saved to '{processed_data_path}'!")
