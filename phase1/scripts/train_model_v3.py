"""
Model Training & Evaluation Pipeline (v3) for SignConnect Live (Phase 1)
------------------------------------------------------------------------
Trains an experimental regularized Neural Network (MLP) on 91-dimensional landmark features
(83 v2 scale-normalized features + 8 3D hand-orientation & palm normal features).
Exports isl_model_v3.joblib and label_map_v3.json.
"""

import os
import csv
import json
import numpy as np
import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

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

def compute_features_v3(pts):
    """
    91-dimensional feature vector:
    - 63 scale-normalized wrist-relative 3D coordinates
    - 10 pairwise tip-to-tip distances
    - 5 fingertip-to-wrist distances
    - 5 finger extension ratios
    - 8 3D Hand Orientation Features:
      * 3D Unit Direction Vector (wrist 0 -> middle MCP 9): (vx, vy, vz)
      * Pitch and Roll orientation angles
      * 3D Unit Palm Normal Vector (cross product of wrist->index and wrist->pinky): (nx, ny, nz)
    """
    wrist = pts[0]
    diffs = pts - wrist
    scale = np.max(np.abs(diffs))
    if scale < 1e-6:
        scale = 1.0

    # 1. 63 scale-normalized coords
    norm_coords = (diffs / scale).flatten()

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    # 2. Pairwise tip-to-tip distances (10)
    tip_dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            tip_dists.append(np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / scale)

    # 3. Tip to wrist distances (5)
    wrist_dists = [np.linalg.norm(pts[t] - wrist) / scale for t in tips]

    # 4. Finger extension ratios (5)
    ext_ratios = []
    for t, p in zip(tips, pips):
        d_t = np.linalg.norm(pts[t] - wrist)
        d_p = np.linalg.norm(pts[p] - wrist) + 1e-6
        ext_ratios.append(d_t / d_p)

    # 5. Hand Orientation Features (8)
    dir_vec = pts[9] - wrist
    dir_norm = np.linalg.norm(dir_vec) + 1e-6
    v_unit = dir_vec / dir_norm

    pitch = np.arctan2(v_unit[1], np.sqrt(v_unit[0]**2 + v_unit[2]**2))
    roll  = np.arctan2(v_unit[0], v_unit[1])

    vec_5  = pts[5] - wrist
    vec_17 = pts[17] - wrist
    normal = np.cross(vec_5, vec_17)
    n_norm = np.linalg.norm(normal) + 1e-6
    n_unit = normal / n_norm

    orient_features = np.array([v_unit[0], v_unit[1], v_unit[2], pitch, roll, n_unit[0], n_unit[1], n_unit[2]])

    return np.concatenate([norm_coords, tip_dists, wrist_dists, ext_ratios, orient_features])

def augment_landmarks_v3(pts):
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

def train_and_evaluate_v3():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("==================================================")
    print("  SignConnect Live - Phase 1 Model Training (v3)  ")
    print("==================================================\n")

    # 1. Load Datasets
    print("Loading raw landmark datasets...")
    X_train_raw, y_train = load_raw_landmarks("train.csv")
    X_val_raw, y_val     = load_raw_landmarks("val.csv")
    X_test_raw, y_test   = load_raw_landmarks("test.csv")

    # 2. Extract 91-dimensional features
    print("Extracting 91-dimensional hand-orientation features...")
    X_train_feat = np.array([compute_features_v3(pts) for pts in X_train_raw])
    X_val_feat   = np.array([compute_features_v3(pts) for pts in X_val_raw])
    X_test_feat  = np.array([compute_features_v3(pts) for pts in X_test_raw])

    # 3. Apply Training Set Augmentation (4x expansion)
    print("Applying training set augmentation (4x)...")
    X_train_aug_list = [X_train_feat]
    y_train_aug_list = [y_train]
    np.random.seed(42)

    for _ in range(3):
        aug_feats = [compute_features_v3(augment_landmarks_v3(pts)) for pts in X_train_raw]
        X_train_aug_list.append(np.array(aug_feats))
        y_train_aug_list.append(y_train)

    X_train_aug = np.vstack(X_train_aug_list)
    y_train_aug = [item for sublist in y_train_aug_list for item in sublist]

    unique_classes = sorted(list(set(y_train + y_val + y_test)))

    # 4. Train Selected Regularized MLP Architecture
    print("Training Regularized MLP v3 (hidden_layer_sizes=(256, 128), alpha=0.10, early_stopping=True)...")
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
    print("               MODEL V3 EVALUATION METRICS        ")
    print("--------------------------------------------------")
    print(f"Train Accuracy:       {train_acc * 100:.2f}%  (Macro F1: {train_f1:.4f}, Weighted F1: {train_wf1:.4f})")
    print(f"Validation Accuracy: {val_acc * 100:.2f}%  (Macro F1: {val_f1:.4f}, Weighted F1: {val_wf1:.4f})")
    print(f"Held-Out Test Acc:   {test_acc * 100:.2f}%  (Macro F1: {test_f1:.4f}, Weighted F1: {test_wf1:.4f})")
    print(f"Generalization Gap:   {gen_gap * 100:.2f}%\n")

    # 6. Save Model & Metadata
    model_path = os.path.join(MODELS_DIR, "isl_model_v3.joblib")
    label_map_path = os.path.join(MODELS_DIR, "label_map_v3.json")

    joblib.dump(clf, model_path)

    metadata = {
        "classes": unique_classes,
        "num_classes": len(unique_classes),
        "num_features": 91,
        "train_accuracy": round(train_acc, 4),
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "test_macro_f1": round(test_f1, 4),
        "test_weighted_f1": round(test_wf1, 4),
        "generalization_gap": round(gen_gap, 4),
        "model_type": "MLPClassifier(256x128, alpha=0.10) + Orientation"
    }

    with open(label_map_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    print(f"V3 Model saved to:    {model_path}")
    print(f"V3 Metadata saved to: {label_map_path}")
    print("\nModel v3 training and evaluation completed successfully!")

if __name__ == "__main__":
    train_and_evaluate_v3()
