"""
Configuration file for fraud detection experiments.
"""
from datetime import datetime

# Dataset configuration
DATA_FILE = "/mnt/1/dissertation/data/prepared/11000_records_balanced.csv"
RUN_DATETIME = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

# Data split configuration
TEST_SIZE = 0.2      # Proportion of data for testing (20%)
VAL_SIZE = 0.25      # Proportion of remaining data for validation (25% of 80% = 20% total)
RANDOM_STATE = 42    # Random seed for reproducibility

# Vectorization configuration
MAX_FEATURES = 5000   # Maximum number of features for TF-IDF (None for unlimited)

# Visualization configuration
SAVE_PLOTS = True               # Whether to save plots to disk
SHOW_PLOTS = False              # Whether to display plots interactively
PLOT_DPI = 100                  # Resolution of saved plots
PLOT_FIGSIZE = (8, 6)          # Default figure size (width, height) in inches

# Experiment configuration
EVALUATE_ON_TRAIN = False  # Whether to evaluate on training set
EVALUATE_ON_VAL = True     # Whether to evaluate on validation set
EVALUATE_ON_TEST = True    # Whether to evaluate on test set
GENERATE_PLOTS = True      # Whether to generate visualization plots

SAVE_MODELS = True  # Whether to save models to disk
SAVE_SUMMARY = True  # Whether to save model summary to disk

# Label encoding
# If labels are 'yes'/'no', they'll be automatically encoded to 1/0
# If they're already numeric, no encoding is needed
POS_LABEL = 1  # The label value for the positive class (fraud)