"""
Model Training & Evaluation Pipeline for SignConnect Live (Phase 1)
-------------------------------------------------------------------
Trains a Random Forest classifier on the 32-class ISL hand landmark dataset,
evaluates performance on held-out test set, and exports the model & label map.
"""

import os
import csv
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FEATURE_COLS = [f"lm{i}_{axis}" for i in range(21) for axis in ('x', 'y', 'z')]

def load_csv_dataset(filename):
    """
    Loads landmark features (X) and string labels (y) from a CSV file.
    """
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    X = []
    y = []

    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                features = [float(row[col]) for col in FEATURE_COLS]
                label = str(row['label']).strip()
                X.append(features)
                y.append(label)
            except (KeyError, ValueError) as e:
                continue

    return X, y

def train_and_evaluate():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("==================================================")
    print("  SignConnect Live - Phase 1 Model Training      ")
    print("==================================================\n")

    # 1. Load Datasets
    print("Loading datasets...")
    X_train, y_train = load_csv_dataset("train.csv")
    X_val, y_val     = load_csv_dataset("val.csv")
    X_test, y_test   = load_csv_dataset("test.csv")

    print(f"Train samples:      {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples:       {len(X_test)}")

    # 2. Get unique classes
    unique_classes = sorted(list(set(y_train + y_val + y_test)))
    print(f"Total ISL Classes: {len(unique_classes)}")
    print(f"Vocabulary: {unique_classes}\n")

    # 3. Train Random Forest Classifier
    print("Training Random Forest Classifier (n_estimators=150, max_depth=15, random_state=42)...")
    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 4. Evaluate Accuracies
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    val_acc   = accuracy_score(y_val, clf.predict(X_val))
    y_test_pred = clf.predict(X_test)
    test_acc  = accuracy_score(y_test, y_test_pred)

    print("\n--------------------------------------------------")
    print("                  EVALUATION RESULTS              ")
    print("--------------------------------------------------")
    print(f"Train Accuracy:      {train_acc * 100:.2f}%")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"Held-Out Test Accuracy: {test_acc * 100:.2f}%\n")

    print("Classification Report on Held-Out Test Set:")
    report = classification_report(y_test, y_test_pred, digits=4)
    print(report)

    # 5. Save Model & Metadata
    model_path = os.path.join(MODELS_DIR, "isl_model.joblib")
    label_map_path = os.path.join(MODELS_DIR, "label_map.json")

    joblib.dump(clf, model_path)
    
    metadata = {
        "classes": unique_classes,
        "num_classes": len(unique_classes),
        "num_features": len(FEATURE_COLS),
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "model_type": "RandomForestClassifier"
    }

    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Model saved to:     {model_path}")
    print(f"Metadata saved to:  {label_map_path}")
    print("\nTraining completed successfully!")

    return metadata

if __name__ == "__main__":
    train_and_evaluate()
