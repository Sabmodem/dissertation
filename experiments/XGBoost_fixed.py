import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading dataset...")
import config; data = pd.read_csv(config.DATA_FILE)

# Split data into train, validation, and test sets
train, temp = train_test_split(data, test_size=0.4, random_state=62)
val, test = train_test_split(temp, test_size=0.5, random_state=62)

# Vectorize the text data using TF-IDF
print("Vectorizing text data...")
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(train['Fillings'])
X_val = tfidf.transform(val['Fillings'])
X_test = tfidf.transform(test['Fillings'])

# Convert labels to numerical format (handle case-insensitive)
y_train = train['Fraud'].str.lower().map({'no': 0, 'yes': 1}).values
y_val = val['Fraud'].str.lower().map({'no': 0, 'yes': 1}).values
y_test = test['Fraud'].str.lower().map({'no': 0, 'yes': 1}).values

# Initialize and train the XGBoost classifier
# In newer XGBoost versions, early_stopping_rounds is passed to the constructor
print("Training XGBoost model...")
clf = xgb.XGBClassifier(
    reg_alpha=0.1,
    reg_lambda=1,
    max_depth=3,
    min_child_weight=1,
    learning_rate=0.3,
    random_state=42,
    early_stopping_rounds=10,  # Moved to constructor in newer versions
    eval_metric="logloss"
)

eval_set = [(X_val, y_val)]
clf.fit(X_train, y_train, eval_set=eval_set, verbose=False)

# Predictions
y_val_pred = clf.predict(X_val)
y_test_pred = clf.predict(X_test)

# Evaluation
print("\n=== Validation Results ===")
print(f"Accuracy: {accuracy_score(y_val, y_val_pred):.4f}")
print(f"Precision: {precision_score(y_val, y_val_pred):.4f}")
print(f"Recall: {recall_score(y_val, y_val_pred):.4f}")
print(f"F1-score: {f1_score(y_val, y_val_pred):.4f}")

print("\n=== Test Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_test_pred):.4f}")
