"""
Configuration file for fraud detection experiments.

Modify this file to point to your dataset and configure experiment settings.
"""

# Dataset configuration
DATA_FILE = "/mnt/1/dissertation-data-balanced/files/Final_Dataset.csv"  # Path to your fraud detection dataset

# Data split configuration
TEST_SIZE = 0.2      # Proportion of data for testing (20%)
VAL_SIZE = 0.25      # Proportion of remaining data for validation (25% of 80% = 20% total)
RANDOM_STATE = 42    # Random seed for reproducibility

# Column names
TEXT_COLUMN = "Fillings"  # Name of the column containing text data
LABEL_COLUMN = "Fraud"    # Name of the column containing labels

# Vectorization configuration
MAX_FEATURES = 5000   # Maximum number of features for TF-IDF (None for unlimited)

# Visualization configuration
PLOTS_DIR = "./plots"           # Directory to save plots
SAVE_PLOTS = True               # Whether to save plots to disk
SHOW_PLOTS = False              # Whether to display plots interactively
PLOT_DPI = 100                  # Resolution of saved plots
PLOT_FIGSIZE = (8, 6)          # Default figure size (width, height) in inches

# Model configuration (examples)
LOGISTIC_REGRESSION_CONFIG = {
    "max_iter": 1000,
    "random_state": RANDOM_STATE,
    "solver": "lbfgs"
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "max_depth": None,
    "min_samples_split": 2
}

SVM_CONFIG = {
    "kernel": "linear",
    "probability": True,
    "random_state": RANDOM_STATE,
    "C": 1.0
}

XGBOOST_CONFIG = {
    "max_depth": 3,
    "learning_rate": 0.3,
    "n_estimators": 100,
    "random_state": RANDOM_STATE,
    "reg_alpha": 0.1,
    "reg_lambda": 1
}

# Neural network configuration (for future use)
KERAS_CONFIG = {
    "layers": [256, 128, 64],
    "activation": "tanh",
    "dropout_rate": 0.0,
    "epochs": 20,
    "batch_size": 32,
    "learning_rate": 0.001,
    "optimizer": "rmsprop",
    "early_stopping_patience": 3
}

# Experiment configuration
EVALUATE_ON_TRAIN = False  # Whether to evaluate on training set
EVALUATE_ON_VAL = True     # Whether to evaluate on validation set
EVALUATE_ON_TEST = True    # Whether to evaluate on test set
GENERATE_PLOTS = True      # Whether to generate visualization plots

# Label encoding
# If your labels are 'yes'/'no', they'll be automatically encoded to 1/0
# If they're already numeric, no encoding is needed
POS_LABEL = 1  # The label value for the positive class (fraud)