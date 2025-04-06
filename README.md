## Data Flow:

Raw Data → Preprocessing: Raw CSV files are read from S3 (mounted as /opt/ml/processing/input) and transformed by the preprocessing script.

Preprocessed Data → Training: The output (processed_train_data.csv and encoder pickle files) is stored back on S3 and later read by the training script.

## SageMaker's Role:

Processing Job: Sets up the container environment (mounting S3 data to local paths) and runs preprocess_train_file_sagemaker.py.

Training Job: Launches an instance (with potential GPU acceleration), downloads preprocessed data from S3, executes train_sagemaker.py, and upon completion, uploads model artifacts and logs to S3.

Pipeline Notebook: The TrainingPipeline.ipynb orchestrates these jobs by configuring input/output S3 paths, job parameters, and monitoring progress through SageMaker’s SDK.

## End-to-End Pipeline:

Preprocessing: Cleans and enriches data, applies text normalization, and encodes categorical variables.

Training: Uses a sophisticated model combining DistilBERT for textual features with embeddings for categorical data, trains with class balancing, and logs performance metrics.

Artifact Management: The best model and related logs are saved and managed in S3 for later evaluation or deployment.
