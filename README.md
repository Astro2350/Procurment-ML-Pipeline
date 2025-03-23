# HierCat-XL
------------------------------------------------

## DistilHier Model - machine learning pipeline to classify financial transactions into hierarchical categories by combining text analysis (using DistilBERT) with structured data. It incrementally learns new categories, evaluates accuracy using metrics and visualizations, and neatly organizes training progress with logs and checkpoints.

------------------------------------------------
Preprocessing Steps:
------------------------------------------------

1. Load Raw Data:

      Take the original AP data and load it into memory.

2. Clean Text Data:

      Remove or correct typos, extra spaces, special characters, or formatting problems.

      Combine multiple text columns (e.g., description, memo) into a single column (combined_text) for easier processing.

3. Encode Categories:

      Transform categorical labels (like "hospital_id", "primary_category_id", or hierarchical categories) into numbers using an encoder.

      Save these numeric mappings (encoders) for later reuse.

4. Handle Numeric Data (like transaction amount):

      Find and adjust extreme numbers (outliers) using methods like clipping (e.g., if a value is extremely high, set it to a reasonable maximum).

------------------------------------------------
Training Steps:
------------------------------------------------

1. Setup and Configuration:

      Load libraries needed for training (PyTorch, transformers, scikit-learn, etc.).

      Set random seeds for reproducibility (so the results are consistent each run).

2. Data Loading:

      Load the preprocessed data into memory.

3. Incremental Learning Setup:

      Load or create a "master label encoder" that tracks all categories the model has seen so far.

      Dynamically adjust the final layer of the neural network if new categories appear.

4. Prepare Data for the Model:

      Initialize a tokenizer (DistilBERT tokenizer) to convert text into numeric tokens.

      Split the data into Training (80%) and Validation (20%) sets.

5. Create PyTorch Dataset:

      Organize the data into batches using a PyTorch Dataset and DataLoader so the model can efficiently learn from batches of data.

6. Define Neural Network (DistilHier Model):

      Build the neural network by combining:

      DistilBERT (turns text into useful numeric patterns).

      Embedding layers (learn numeric representations of categorical features).

      Fully connected layers (learn patterns combining text, categorical, hierarchical, and numeric features like transaction amounts).

      ReLU activation (helps model learn faster).

      Batch Normalization and Dropout (help prevent overfitting).

7. Set Optimization & Training Strategies:

      Define loss function (CrossEntropy with class weights) to tell the model how to improve predictions.

      Use AdamW Optimizer to quickly find optimal model parameters.

      Use ReduceLROnPlateau (learning rate scheduler) to automatically adjust learning speed.

8. Training Loop:

      For multiple epochs (passes through the entire dataset):

      Model sees training data, makes predictions, measures errors, and adjusts its parameters to improve.

      After each epoch, the model evaluates itself on unseen validation data to ensure it's generalizing well.

9. Early Stopping:

      Monitor model performance. Stop early if validation performance stops improving to save time and prevent overfitting.

10. Save Best Model & Checkpoints:

      Save the model whenever it reaches new best performance.

      Store checkpoints (saved models) to resume training later or to deploy for predictions.

11. Visualization & Logging:

      Generate confusion matrices (visual charts showing model accuracy).

      Log metrics (accuracy, loss, precision, recall, F1-score) to files and TensorBoard for easy monitoring.

------------------------------------------------

Methods/Techniques:

      Batch Normalization

      Dropout

      Early Stopping

      ReLU Activation

Optimization Algorithms:

      AdamW Optimizer

      Learning Rate Scheduler (ReduceLROnPlateau)

Implementation Techniques:

      Efficient Data Loading (DataLoader)

Architecture & Incremental Learning Strategies:

      Dynamic Model Adjustment
