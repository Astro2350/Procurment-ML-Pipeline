# CT-Train
------------------------------------------------
# What does each level in a category describe?
------------------------------------------------

## Hierarchical Category Model - combines DistilBERT’s text embeddings with learned embeddings for structured categorical and hierarchical features to classify transactions into a detailed multi-level category tree

------------------------------------------------
Preprocessing Steps:
------------------------------------------------

Data Loading and Cleaning:
Load the raw CSV files (one for the category hierarchy and one for training data - consists of the usual ap_row and vendors' primary category).
Standardize column names to a consistent format.

Data Merging and Feature Engineering:
Merge the training data with the categories data using a common key (category id).
Combine text fields (like name, description, and memo) into a single text feature.
Apply text cleaning—converting text to lowercase and removing unwanted characters. (Previously, stopwords removal and lemmatization were applied, but those are emitted in this setup - HCA has fairly clean data.)

Target Variable Processing:
Encode the target labels (matched categories) into numerical format.
Filter out classes that have only one sample to avoid issues during training.
Re‑encode the filtered labels so they form a continuous range.

Saving Processed Data:
Save the processed dataset and the fitted label encoder to the working directory so that the training script can use them without re‑processing.

------------------------------------------------
Training Steps:
------------------------------------------------

Loading Processed Data:
Read the processed data and label encoder from the working directory.

Data Splitting and Dataset Creation:
Split the processed data into training and validation sets.
Create a custom dataset class that tokenizes the combined text (using a pre-trained tokenizer) and prepares the additional features (like categorical and hierarchical features, and a numerical amount).

Model Setup:
Load a pre-trained transformer model (such as DistilBERT) along with its tokenizer.
Build a deep learning model that combines:
The transformer’s output (for text features).
Additional embeddings for other categorical features.
Fully connected layers that integrate all features and produce predictions for the target classes.

Training Configuration:
Configure an optimizer (like AdamW), compute class weights to handle imbalanced data, and set up a learning rate scheduler.
Optionally, set up mixed precision training if a GPU is available (....it definitley will be with AWS Sagemaker but not this run through).

Training Loop:
Iterate over the training data in batches, feed the data through the model, calculate loss (using cross-entropy), and update model weights.
Monitor performance on the validation set and implement early stopping if necessary.
Save model checkpoints and logs for tracking progress.

Saving Final Artifacts:
After training, save the best model and the category mapping.
