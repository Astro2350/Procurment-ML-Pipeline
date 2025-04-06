import pandas as pd
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Function to preprocess text (same as in training)
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
    return text

# Define base path
base_path = r"C:\Users\sambe\Desktop\ML Stuff\CT-Train"

# Load raw test data and list_of_categories (if needed for merging)
test_data_path = os.path.join(base_path, 'TEST.csv')
test_df = pd.read_csv(test_data_path)
categories_df = pd.read_csv(os.path.join(base_path, 'list_of_categories.csv'))

print("Preprocessing test data...")

# Clean column names for consistency
test_df.columns = test_df.columns.str.strip().str.lower().str.replace(' ', '_')
categories_df.columns = categories_df.columns.str.strip().str.lower().str.replace(' ', '_')

# Merge test data with category information (if required)
merged_df = test_df.merge(categories_df, left_on='primary_category_id', right_on='id', how='left')

# Create combined_text field after filling missing values
merged_df['name'] = merged_df['name'].fillna('')
merged_df['gl_description'] = merged_df['gl_description'].fillna('')
merged_df['memo'] = merged_df['memo'].fillna('')
merged_df['combined_text'] = merged_df['name'] + ' ' + merged_df['gl_description'] + ' ' + merged_df['memo']
merged_df['combined_text'] = merged_df['combined_text'].apply(preprocess_text)

# ---- Load saved feature encoders from training ----
with open(os.path.join(base_path, 'hospital_system_encoder.pkl'), 'rb') as f:
    hospital_encoder = pickle.load(f)
with open(os.path.join(base_path, 'department_encoder.pkl'), 'rb') as f:
    department_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category0_encoder.pkl'), 'rb') as f:
    category0_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category1_encoder.pkl'), 'rb') as f:
    category1_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category2_encoder.pkl'), 'rb') as f:
    category2_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category3_encoder.pkl'), 'rb') as f:
    category3_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category4_encoder.pkl'), 'rb') as f:
    category4_encoder = pickle.load(f)
with open(os.path.join(base_path, 'category5_encoder.pkl'), 'rb') as f:
    category5_encoder = pickle.load(f)

# ---- Apply the saved encoders to create encoded columns ----
merged_df['hospital_system_id'] = merged_df['hospital_system_id'].astype(str)
merged_df['hospital_system_id_encoded'] = hospital_encoder.transform(merged_df['hospital_system_id'])

merged_df['department_name'] = merged_df['department_name'].fillna('Unknown')
merged_df['department_name_encoded'] = department_encoder.transform(merged_df['department_name'])

merged_df['category0'] = merged_df['category0'].fillna('Unknown')
merged_df['category0_encoded'] = category0_encoder.transform(merged_df['category0'])

merged_df['category1'] = merged_df['category1'].fillna('Unknown')
merged_df['category1_encoded'] = category1_encoder.transform(merged_df['category1'])

merged_df['category2'] = merged_df['category2'].fillna('Unknown')
merged_df['category2_encoded'] = category2_encoder.transform(merged_df['category2'])

merged_df['category3'] = merged_df['category3'].fillna('Unknown')
merged_df['category3_encoded'] = category3_encoder.transform(merged_df['category3'])

merged_df['category4'] = merged_df['category4'].fillna('Unknown')
merged_df['category4_encoded'] = category4_encoder.transform(merged_df['category4'])

merged_df['category5'] = merged_df['category5'].fillna('Unknown')
merged_df['category5_encoded'] = category5_encoder.transform(merged_df['category5'])

# ---- Add primary_category_id_encoded if not already present ----
if 'primary_category_id_encoded' not in merged_df.columns:
    primary_encoder_path = os.path.join(base_path, 'primary_category_encoder.pkl')
    try:
        with open(primary_encoder_path, 'rb') as f:
            primary_encoder = pickle.load(f)
        merged_df['primary_category_id_encoded'] = primary_encoder.transform(merged_df['primary_category_id'])
    except Exception as e:
        print("Primary category encoder not found or error in transformation:", e)

# Save the processed test data for inference
processed_test_data_path = os.path.join(base_path, 'processed_test_data.csv')
merged_df.to_csv(processed_test_data_path, index=False)

print(f"Test data preprocessing complete and saved to '{processed_test_data_path}'!")
