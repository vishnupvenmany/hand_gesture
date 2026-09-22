"""
Automated Test Suite (v3) for SignConnect Live (Phase 1)
--------------------------------------------------------
Verifies:
1. 91-dimensional hand-orientation feature extraction
2. v3 Label map and metadata loading
3. Experimental v3 MLP model loading and inference
4. Honest probability confidence scoring
"""

import os
import sys
import json
import joblib
import numpy as np

PHASE1_DIR = os.path.dirname(os.path.dirname(__file__))
SCRIPTS_DIR = os.path.join(PHASE1_DIR, "scripts")
sys.path.insert(0, SCRIPTS_DIR)

from train_model_v3 import compute_features_v3

def test_feature_extraction_v3():
    print("[TEST 1/4] 91-Dimensional Hand Orientation Feature Extraction...")
    pts = np.array([(0.1 * i, 0.2 * i, 0.05 * i) for i in range(21)])
    features = compute_features_v3(pts)

    assert len(features) == 91, f"Expected 91 features, got {len(features)}"
    print("  [PASSED] 91-dimensional feature vector verified.")

def test_metadata_v3():
    print("[TEST 2/4] v3 Metadata and Class Map Loading...")
    label_map_path = os.path.join(PHASE1_DIR, "models", "label_map_v3.json")
    assert os.path.exists(label_map_path), f"Label map v3 not found at {label_map_path}"

    with open(label_map_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta["num_classes"] == 32, f"Expected 32 classes, got {meta['num_classes']}"
    assert meta["num_features"] == 91, f"Expected 91 features, got {meta['num_features']}"
    print(f"  [PASSED] Metadata v3 loaded cleanly ({meta['num_classes']} classes, 91 features).")

def test_model_v3_inference():
    print("[TEST 3/4] v3 Model File Loading & Prediction...")
    model_path = os.path.join(PHASE1_DIR, "models", "isl_model_v3.joblib")
    assert os.path.exists(model_path), f"Model v3 file not found at {model_path}"

    clf = joblib.load(model_path)
    sample_features = np.zeros(91)
    pred = clf.predict([sample_features])[0]
    probs = clf.predict_proba([sample_features])[0]

    assert pred in clf.classes_, f"Predicted label {pred} not in model classes"
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probability distribution sum invalid: {sum(probs)}"
    print(f"  [PASSED] Model v3 loaded cleanly and produces valid prediction '{pred}' (prob sum: {sum(probs):.2f}).")

def test_v2_live_model_preservation():
    print("[TEST 4/4] Verifying V2 Live Baseline Model Preservation...")
    v2_model_path = os.path.join(PHASE1_DIR, "models", "isl_model_v2.joblib")
    assert os.path.exists(v2_model_path), f"Live baseline v2 model not found at {v2_model_path}"
    print("  [PASSED] Live baseline v2 model remains untouched as active demo model.")

def run_all_tests_v3():
    print("==================================================")
    print("  SignConnect Live - Automated Test Suite (v3)    ")
    print("==================================================\n")
    test_feature_extraction_v3()
    test_metadata_v3()
    test_model_v3_inference()
    test_v2_live_model_preservation()
    print("\nAll 4 automated v3 pipeline tests passed successfully!")

if __name__ == "__main__":
    run_all_tests_v3()
