## Step 1: Imports & Setup
What this step does:
This step brings in the tools (libraries) that the script will use to perform tasks like data processing, math operations, and machine learning.

Key Libraries & What They Do:
Pandas (pandas):

Loads, reads, and manipulates structured data, similar to Excel but in Python.

NumPy (numpy):

Performs mathematical calculations quickly, especially on large sets of numbers.

PyTorch (torch):

A powerful library used to build, train, and run deep learning models.

Transformers (from Hugging Face):

Offers advanced pretrained AI models (like DistilBERT) to process and understand text data.

Scikit-learn (sklearn):

Provides tools for splitting data, encoding labels, calculating metrics, and handling class imbalance.

TensorBoard (torch.utils.tensorboard):

A visualization toolkit from Google that tracks and displays graphs of how well the model learns during training.

Matplotlib & Seaborn (matplotlib, seaborn):

Used to create visual charts (like confusion matrices) that help see model performance visually.

## Step 2: Directory & Experiment Management
What this step does:
Creates folders to neatly organize your training results, models, and logs each time you run the script. This way, you won't overwrite previous results and can easily compare performance over multiple runs.

## Step 3: TensorBoard Initialization
What TensorBoard is:
TensorBoard is a dashboard-like tool that shows graphs, charts, and statistics about your model’s training performance (accuracy, loss, etc.), updated live as your model trains.

## Step 4: Incremental Learning Check
Incremental Learning:
Incremental learning means your model learns new things (like new categories) without forgetting previous ones. The script looks at past experiments to find previous knowledge (saved models) to build upon.

## Step 5: Data Loading & Master Label Encoder
What Label Encoding Means:
Label Encoding converts text labels (categories like "Food", "Office Supplies", etc.) into numeric codes (0, 1, 2, etc.) because the model needs numbers to make predictions.

The master label encoder ensures the numeric codes for categories remain consistent across different training sessions.

## Step 6: Updating Master Label Encoder
Why Update the Encoder:
When new data arrives, it might have new categories that didn’t exist before. The encoder updates itself to recognize and handle these new categories.

## Step 7: Data Preprocessing & Outlier Handling
What this means:
Preprocessing cleans the data before training. One important part of cleaning is managing extreme or unusual numbers (outliers). For example, if most transactions are around $100 but suddenly one says $1,000,000, this step either trims it down or adjusts it to avoid confusing the model.

## Step 8: Tokenizer Initialization
What Tokenizer & DistilBERT mean:
A tokenizer splits text into smaller chunks called tokens, which makes it easier for AI models to analyze.

DistilBERT is a pretrained AI model (a Transformer) that understands language. It has already read and learned from massive amounts of text online, making it good at converting text into meaningful numbers (embeddings) the model can use.

## Step 9: Custom PyTorch Dataset
Why Custom Dataset:
The script uses a custom-defined dataset that neatly organizes the text data, categorical data (like hospital IDs), hierarchical categories, and numerical data (like transaction amounts). It prepares all these data points so the model can process them together smoothly.

## Step 10: Neural Network Model Definition
Neural Network Basics:
A neural network is like a complex web of math operations inspired by human brains, able to learn patterns from data.

The Layers Explained:
DistilBERT Layer:

Converts your input text (like descriptions) into numeric representations called embeddings.

Embedding Layers (for categorical and hierarchical features):

Convert categories (like "hospital_id" or "primary_category_id") into numeric embeddings that the model can interpret.

Fully Connected Layers (fc1, fc2, fc3):

These layers combine all embeddings and numerical data into predictions. They work by finding mathematical patterns that map inputs to categories.

Batch Normalization (BatchNorm) & Dropout:

Batch Normalization keeps numbers stable to help the model learn faster.

Dropout temporarily removes random connections between neurons during training to avoid overfitting (memorizing data instead of learning general patterns).

## Step 11: Adjusting Model for New Classes
Dynamic Adjustments:
If new categories appear, the final layer of the model adjusts to accommodate these categories. It preserves existing learned weights to prevent loss of previous knowledge.

## Step 12: Class Weights Computation
What Class Weights Are:
In datasets, some categories might appear very frequently while others rarely appear. Class weights balance this, giving more importance to rare categories so the model learns them equally well.

## Step 13: Train/Validation Data Split & Loaders
What this does:
The data is divided into two sets:

Training set: the model learns patterns from this.

Validation set: the model tests its learned patterns on unseen data to check generalization.

Data loaders feed data batches to the model efficiently during training.

## Step 14: Hyperparameters & Optimization
Hyperparameters Explained:
Learning rate: How quickly the model adjusts its understanding of patterns.

Optimizer (AdamW): Helps the model adjust efficiently.

Loss function (CrossEntropy): Measures how wrong the model predictions are to help improve accuracy.

## Step 15: Loading Checkpoints
Checkpoints:
If training was paused or done previously, checkpoints store the model’s learned state so it can resume training later without starting over.

## Step 16: Training & Evaluation Functions
Training Function:
This function repeatedly feeds data batches to the model, calculates the error (loss), and adjusts the model weights to minimize this error.

Evaluation Function:
Checks how accurately the trained model predicts on unseen data.

## Step 17: Confusion Matrix Visualization
Confusion Matrix:
A visual chart showing clearly what categories the model predicts correctly and where it gets confused, helping you understand performance at a glance.

## Step 18: Main Training Loop
What this loop does:
Repeats the training and evaluation process multiple times (epochs), continuously improving the model.

Saves the best-performing version of the model based on validation performance (measured by F1-score).

Stops early if the model stops improving (early stopping).

## Step 19: Post-Training Logging
Completes training by neatly saving logs, metrics, model checkpoints, and all artifacts generated during training.
