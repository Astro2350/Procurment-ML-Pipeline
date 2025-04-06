import os
import re
import pickle
import pandas as pd

# Minimal hard-coded stop words (if desired)
stop_words = {
    'the', 'and', 'is', 'in', 'to', 'of', 'for', 'on',
    'with', 'at', 'by', 'an', 'be', 'this', 'from'
}

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetic chars
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove simple stopwords
    words = [w for w in text.split() if w not in stop_words]
    return ' '.join(words)

# SageMaker input/output paths
input_path = "/opt/ml/processing/input"
output_path = "/opt/ml/processing/output"

try:
    print("Reading test CSV and categories CSV from input...")
    
    # Typically your pipeline step sets:
    #   destination="/opt/ml/processing/input/test" for TEST.csv
    #   destination="/opt/ml/processing/input/categories" for list_of_categories.csv
    test_csv = os.path.join(input_path, "test", "TEST.csv")
    categories_csv = os.path.join(input_path, "categories", "list_of_categories.csv")
    
    test_df = pd.read_csv(test_csv, engine="python")
    categories_df = pd.read_csv(categories_csv, engine="python")
    
    print(f"Loaded test data rows: {len(test_df)}")
    print(f"Loaded categories rows: {len(categories_df)}")
    
    # Clean column names
    test_df.columns = test_df.columns.str.strip().str.lower().str.replace(' ', '_')
    categories_df.columns = categories_df.columns.str.strip().str.lower().str.replace(' ', '_')
    
    print("Merging test data with categories info (if needed)...")
    merged_df = test_df.merge(categories_df, left_on='primary_category_id', right_on='id', how='left')
    
    # Fill missing text fields
    merged_df['name'] = merged_df.get('name', '').fillna('')
    merged_df['gl_description'] = merged_df.get('gl_description', '').fillna('')
    merged_df['memo'] = merged_df.get('memo', '').fillna('')
    merged_df['department_name'] = merged_df.get('department_name', '').fillna('')
    
    # Combine into one text field, then preprocess
    merged_df['combined_text'] = (
        merged_df['name'] + ' ' +
        merged_df['gl_description'] + ' ' +
        merged_df['memo'] + ' ' +
        merged_df['department_name']
    ).apply(preprocess_text)
    
    print("Loading encoders from training (if needed)...")
    # If your pipeline step sets:
    #   destination="/opt/ml/processing/input/encoders" for .pkl files
    # then you can read them like:
    encoders_dir = os.path.join(input_path, "encoders")
    
    hospital_encoder_path = os.path.join(encoders_dir, "hospital_system_encoder.pkl")
    department_encoder_path = os.path.join(encoders_dir, "department_encoder.pkl")
    category0_encoder_path = os.path.join(encoders_dir, "category0_encoder.pkl")
    # ... likewise for category1..category5
    # ... possibly primary_category_encoder.pkl if needed
    
    # If you don't have these or don't need them, remove or wrap in try/except
    # Example usage:
    if os.path.isfile(hospital_encoder_path):
        with open(hospital_encoder_path, 'rb') as f:
            hospital_encoder = pickle.load(f)
        merged_df['hospital_system_id'] = merged_df['hospital_system_id'].astype(str)
        merged_df['hospital_system_id_encoded'] = hospital_encoder.transform(merged_df['hospital_system_id'])
    
    if os.path.isfile(department_encoder_path):
        with open(department_encoder_path, 'rb') as f:
            department_encoder = pickle.load(f)
        merged_df['department_name'] = merged_df['department_name'].fillna('Unknown')
        merged_df['department_name_encoded'] = department_encoder.transform(merged_df['department_name'])
    
    if os.path.isfile(category0_encoder_path):
        with open(category0_encoder_path, 'rb') as f:
            category0_encoder = pickle.load(f)
        merged_df['category0'] = merged_df['category0'].fillna('Unknown')
        merged_df['category0_encoded'] = category0_encoder.transform(merged_df['category0'])
    
    # Repeat for category1..5 if you have them
    # ...
    
    # If you have a primary_category_encoder.pkl, do similarly
    # ...
    
    # Save processed test data to output
    output_csv = os.path.join(output_path, "processed_test_data.csv")
    merged_df.to_csv(output_csv, index=False)
    print(f"Processed test data saved to: {output_csv}")

except Exception as e:
    print(f"Error preprocessing test data: {e}")
    import traceback
    traceback.print_exc()
    raise
