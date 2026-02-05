import os
# Force TensorFlow to use CPU only to avoid GPU/CUDA issues
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix

# Load the data
import config; df = pd.read_csv(config.DATA_FILE)

# Convert labels to binary
le = LabelEncoder()
df['Fraud'] = le.fit_transform(df['Fraud'])

# Split data into train, validation, and test sets
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train = vectorizer.fit_transform(train['Fillings']).toarray()
X_val = vectorizer.transform(val['Fillings']).toarray()
X_test = vectorizer.transform(test['Fillings']).toarray()

y_train = train['Fraud'].values
y_val = val['Fraud'].values
y_test = test['Fraud'].values

# Define the ANN Model with Input layer (compatible with newer Keras)
model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Model
early_stop = EarlyStopping(monitor='val_loss', patience=3)

print("Training main model...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop],
    verbose=1
)

# Evaluate the Model
# Validation performance
y_val_pred_prob = model.predict(X_val, verbose=0)
y_val_pred = (y_val_pred_prob > 0.5).astype("int32")
val_accuracy = accuracy_score(y_val, y_val_pred)
val_precision = precision_score(y_val, y_val_pred)
val_recall = recall_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred)

# Test performance
y_test_pred_prob = model.predict(X_test, verbose=0)
y_test_pred = (y_test_pred_prob > 0.5).astype("int32")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"\n=== Main Model (Tanh) Results ===")
print(f"Validation Accuracy: {val_accuracy:.4f}")
print(f"Validation Precision: {val_precision:.4f}")
print(f"Validation Recall: {val_recall:.4f}")
print(f"Validation F1-score: {val_f1:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1-score: {test_f1:.4f}")

# Define and Train the Model (ReLU Activation)
print("\nTraining ReLU model...")
model_relu = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_relu.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history_relu = model_relu.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop],
    verbose=1
)

# Define and Train the Model (Tanh Activation)
print("\nTraining Tanh model...")
model_tanh = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(256, activation='tanh'),
    Dense(128, activation='tanh'),
    Dense(64, activation='tanh'),
    Dense(1, activation='sigmoid')
])
model_tanh.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

history_tanh = model_tanh.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_data=(X_val, y_val), 
    callbacks=[early_stop],
    verbose=1
)

# Evaluate and Plot Graphs
def evaluate_and_plot(model, X_test, y_test, model_name):
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"\n=== {model_name} Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Fraudulent", "Fraudulent"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=model_name)
    roc_display.plot()
    plt.title(f'ROC Curve - {model_name}')
    plt.savefig(f'roc_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

    # Plot Precision-Recall Curve
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_prob)
    pr_display = PrecisionRecallDisplay(precision=precision_curve, recall=recall_curve)
    pr_display.plot()
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.savefig(f'pr_curve_{model_name.lower().replace(" ", "_")}.png')
    plt.close()

print("\n" + "="*50)
evaluate_and_plot(model_relu, X_test, y_test, "ReLU Activation")
evaluate_and_plot(model_tanh, X_test, y_test, "Tanh Activation")

print("\n" + "="*50)
print("Training complete! Plots saved as PNG files.")
