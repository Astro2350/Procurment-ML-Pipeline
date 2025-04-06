# Overall Pipeline in SageMaker
Orchestration via the Training Pipeline Notebook
The Jupyter Notebook (TrainingPipeline.ipynb) serves as the high-level orchestrator. It likely uses the SageMaker SDK to:

Launch a Processing Job: Runs the preprocessing script (preprocess_train_file_sagemaker.py) to clean and transform raw data.

Launch a Training Job: Invokes the training script (train_sagemaker.py) that trains the deep learning model.

Manage Data and Artifacts: It specifies S3 buckets/paths so that processed data and model artifacts (like checkpoints, logs, and reports) are automatically stored in S3.

# SageMaker Role
## SageMaker manages the compute environments:

Processing Job Environment: Maps S3 input data to /opt/ml/processing/input and later saves outputs from /opt/ml/processing/output back to S3.

Training Job Environment: Sets up the container (possibly with GPU support), downloads processed data from S3, runs the training script, and then uploads the experiment’s outputs (trained model, logs, reports) to S3.

Preprocessing Step: preprocess_train_file_sagemaker.py
This script is executed as a SageMaker processing job. Its detailed operations are:

## Data Loading & Setup:

Input Paths: It reads CSV files (e.g., TRAIN.csv and list_of_categories.csv) from /opt/ml/processing/input. These paths are provided by SageMaker based on the S3 input configuration.

Library Initialization: It downloads NLTK resources (stopwords and WordNet lemmatizer) to ensure text processing dependencies are met.

## Data Cleaning and Merging:

Column Standardization: Column names in both dataframes are standardized (stripped, lowercased, underscores replacing spaces).

Merging: The training data is merged with category information (joining on identifiers) to enrich the dataset.

## Text Preprocessing:

Combined Text Field: Several text columns (name, gl_description, memo, department_name) are concatenated into a single combined_text field.

Preprocessing Function: The preprocess_text function converts text to lowercase, removes non-alphabet characters via regex, removes stopwords, and applies lemmatization. This function is applied to the combined_text column for uniform text processing.

## Label Encoding & Filtering:

Encoding: Categorical columns (like matched_category_id, primary_category_id, hospital_system_id, and others) are encoded using scikit-learn’s LabelEncoder.

Rare Class Removal: Any classes that occur only once are filtered out, and the target labels are re-encoded to ensure a consistent label space.

## Saving Outputs:

Processed Data: The final preprocessed DataFrame is saved as processed_train_data.csv in /opt/ml/processing/output.

Encoders: The fitted label encoders are pickled (saved as .pkl files) to ensure that the same transformations can be applied during inference or by the training job.

This entire preprocessing workflow is detailed in the script (see ).

Training Step: train_sagemaker.py
This script is run as a SageMaker training job. Its key technical operations include:

## Environment and Data Loading:

S3 Integration: The script defines a base S3 path (e.g., s3://lxeml/CH_Test/) from which it loads the preprocessed data (processed_train_data.csv) and the master label encoder. SageMaker’s training job environment ensures that the required data is accessible via these S3 URIs.

Device Setup: It detects the availability of GPU (via torch.cuda.is_available()) to set the device appropriately.

## Preliminary Data Handling:

Outlier Treatment: There is a function to handle outliers in numeric columns (specifically amount), applying clipping based on the interquartile range (IQR).

Tokenization: A DistilBERT tokenizer is initialized to convert the combined text into token IDs that the model can process.

## Dataset and Model Preparation:

Custom Dataset: A PyTorch Dataset class (TransactionDataset) is defined. It tokenizes each row’s combined_text and packs in additional features:

Text Features: Tokenized input IDs and attention masks.

Categorical Features: Encoded categorical features such as hospital_system_id_encoded and primary_category_id_encoded.

Hierarchical Features: Encoded values for several hierarchical categories.

Target Label: The encoded matched_category_id.

## Model Architecture:

BERT Backbone: The model uses DistilBERT to extract contextual text embeddings (using the [CLS] token output).

Embedding Layers: Separate embedding layers are created for the additional categorical and hierarchical features.

Fully Connected Layers: These embeddings, along with a raw numeric feature (amount), are concatenated with the BERT output and fed through several linear layers with dropout and batch normalization.

Output: The final layer produces logits corresponding to the number of classes.

These details of model definition and data handling are described in .

Training Loop and Optimization

## Training Setup:

Random Seed: For reproducibility, a fixed seed is set.

Data Splitting: The preprocessed data is split into training and validation sets.

Oversampling: To balance classes, the training data is oversampled using WeightedRandomSampler.

## Loss and Optimizer:

Loss Function: Cross-entropy loss is used, with class weights computed to address imbalance.

Optimizer: AdamW is used with a learning rate of 1e-5.

## Epoch Loop:

In each epoch, the training loop iterates over batches, computes gradients, applies gradient clipping, and updates weights.

Validation: After each epoch, validation metrics (loss, accuracy, F1, top-N accuracy) are computed.

Logging: Metrics are logged to TensorBoard and a CSV log file is maintained.

Early Stopping & Checkpointing: The best model (based on F1 score) is saved as a checkpoint (best_model.pth) in the experiment directory.

Visualization: A histogram of prediction confidences is saved to monitor model calibration.

## Saving Artifacts:

Model Artifacts: The best model checkpoint and optimizer state are saved in a timestamped experiment directory.

S3 Upload: When the training job concludes, SageMaker automatically uploads the experiment directory (including model checkpoints, logs, and reports) to the specified S3 bucket as part of the job’s output.

The training script’s detailed mechanics are fully captured in .
