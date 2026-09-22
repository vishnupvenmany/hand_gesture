"""
Model Training & Evaluation Pipeline (v2) for SignConnect Live (Phase 1)
------------------------------------------------------------------------
Trains an improved regularized Neural Network (MLP) on 83-dimensional landmark features,
applies safe training set augmentation, evaluates performance on held-out test data,
and exports isl_model_v2.joblib and label_map_v2.json.
"""

import os
import csv
import json
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

try:
    from phase1.feature_extractor import extract_features_from_numpy
except ImportError:
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "phase1"))
    from feature_extractor import extract_features_from_numpy

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURE_COLS = [f"lm{i}_{axis}" for i in range(21) for axis in ('x', 'y', 'z')]

def load_raw_landmarks(filename):
    filepath = os.path.join(RAW_DATA_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    X_pts, y = [], []
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                coords = [float(row[col]) for col in FEATURE_COLS]
                label = str(row['label']).strip()
                pts = np.array(coords).reshape(21, 3)
                X_pts.append(pts)
                y.append(label)
            except (KeyError, ValueError):
                continue
    return X_pts, y

def augment_landmarks(pts):
    """Applies small 2D rotation, scaling, and Gaussian jitter to training landmarks."""
    angle = np.radians(np.random.uniform(-10, 10))
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    rot_mat = np.array([
        [cos_a, -sin_a, 0],
        [sin_a,  cos_a, 0],
        [    0,      0, 1]
    ])
    scale = np.random.uniform(0.95, 1.05)
    noise = np.random.normal(0, 0.005, pts.shape)
    return np.dot(pts, rot_mat.T) * scale + noise

def train_and_evaluate_v2():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("==================================================")
    print("  SignConnect Live - Phase 1 Model Training (v2)  ")
    print("==================================================\n")

    # 1. Load Datasets
    print("Loading raw landmark datasets...")
    X_train_raw, y_train = load_raw_landmarks("train.csv")
    X_val_raw, y_val     = load_raw_landmarks("val.csv")
    X_test_raw, y_test   = load_raw_landmarks("test.csv")

    # 2. Extract 83-dimensional features
    print("Extracting 83-dimensional scale-normalized features...")
    X_train_feat = np.array([extract_features_from_numpy(pts) for pts in X_train_raw])
    X_val_feat   = np.array([extract_features_from_numpy(pts) for pts in X_val_raw])
    X_test_feat  = np.array([extract_features_from_numpy(pts) for pts in X_test_raw])

    # 3. Apply Training Set Augmentation (4x dataset expansion strictly on train)
    print("Applying physically plausible training data augmentation (4x)...")
    X_train_aug_list = [X_train_feat]
    y_train_aug_list = [y_train]
    np.random.seed(42)

    for _ in range(3):
        aug_feats = [extract_features_from_numpy(augment_landmarks(pts)) for pts in X_train_raw]
        X_train_aug_list.append(np.array(aug_feats))
        y_train_aug_list.append(y_train)

    X_train_aug = np.vstack(X_train_aug_list)
    y_train_aug = [item for sublist in y_train_aug_list for item in sublist]

    print(f"Augmented Train samples: {len(X_train_aug)}")
    print(f"Validation samples:      {len(X_val_feat)}")
    print(f"Test samples:            {len(X_test_feat)}")

    unique_classes = sorted(list(set(y_train + y_val + y_test)))
    print(f"Total ISL Classes:       {len(unique_classes)}\n")

    # 4. Train Selected Regularized MLP Architecture
    print("Training Regularized MLP (hidden_layer_sizes=(256, 128), alpha=0.10, early_stopping=True)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        max_iter=600,
        alpha=0.10,
        early_stopping=True,
        random_state=42
    )
    clf.fit(X_train_aug, y_train_aug)

    # 5. Evaluate Accuracies & F1 Scores
    y_train_pred = clf.predict(X_train_aug)
    y_val_pred   = clf.predict(X_val_feat)
    y_test_pred  = clf.predict(X_test_feat)

    train_acc = accuracy_score(y_train_aug, y_train_pred)
    val_acc   = accuracy_score(y_val, y_val_pred)
    test_acc  = accuracy_score(y_test, y_test_pred)

    train_f1  = f1_score(y_train_aug, y_train_pred, average='macro')
    val_f1    = f1_score(y_val, y_val_pred, average='macro')
    test_f1   = f1_score(y_test, y_test_pred, average='macro')

    train_wf1 = f1_score(y_train_aug, y_train_pred, average='weighted')
    val_wf1   = f1_score(y_val, y_val_pred, average='weighted')
    test_wf1  = f1_score(y_test, y_test_pred, average='weighted')

    gen_gap   = train_acc - test_acc

    print("\n--------------------------------------------------")
    print("            IMPROVED V2 EVALUATION METRICS        ")
    print("--------------------------------------------------")
    print(f"Train Accuracy:       {train_acc * 100:.2f}%  (Macro F1: {train_f1:.4f}, Weighted F1: {train_wf1:.4f})")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%  (Macro F1: {val_f1:.4f}, Weighted F1: {val_wf1:.4f})")
    print(f"Held-Out Test Acc:   {test_acc * 100:.2f}%  (Macro F1: {test_f1:.4f}, Weighted F1: {test_wf1:.4f})")
    print(f"Generalization Gap:   {gen_gap * 100:.2f}%\n")

    print("Classification Report on Held-Out Test Set:")
    report = classification_report(y_test, y_test_pred, digits=4)
    print(report)

    # 6. Save Model & Metadata
    model_path = os.path.join(MODELS_DIR, "isl_model_v2.joblib")
    label_map_path = os.path.join(MODELS_DIR, "label_map_v2.json")

    joblib.dump(clf, model_path)

    metadata = {
        "classes": unique_classes,
        "num_classes": len(unique_classes),
        "num_features": 83,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_weighted_f1": round(test_wf1, 4),
        "generalization_gap": round(gen_gap, 4),
        "model_type": "MLPClassifier(256x128, alpha=0.10)"
    }

    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"Improved model saved to:    {model_path}")
    print(f"Improved metadata saved to: {label_map_path}")
    print("\nModel v2 training and evaluation completed successfully!")

if __name__ == "__main__":
    train_and_evaluate_v2()
