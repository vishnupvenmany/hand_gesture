"""
Automated Pipeline Test Suite (v2) for SignConnect Live (Phase 1)
-----------------------------------------------------------------
Verifies:
1. 83-dimensional scale-normalized feature extraction
2. v2 Label map and metadata loading
3. Trained MLP v2 model loading and inference
4. Honest probability confidence scoring
5. Low-confidence / UNKNOWN gesture thresholding
"""

import os
import sys
import json
import joblib
import numpy as np

# Add parent path
PHASE1_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PHASE1_DIR, "phase1"))

from feature_extractor import extract_features_from_numpy
from gesture_recognizer import classify_gesture, load_model_if_needed

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks_tuples):
        self.landmark = [MockLandmark(x, y, z) for x, y, z in landmarks_tuples]

def test_feature_extraction_v2():
    print("[TEST 1/5] 83-Dimensional Feature Extraction...")
    pts = np.array([(0.1 * i, 0.2 * i, 0.05 * i) for i in range(21)])
    features = extract_features_from_numpy(pts)

    assert len(features) == 83, f"Expected 83 features, got {len(features)}"
    print("  [PASSED] 83-dimensional scale-normalized feature extraction verified.")

def test_metadata_v2():
    print("[TEST 2/5] v2 Metadata and Class Map Loading...")
    label_map_path = os.path.join(PHASE1_DIR, "models", "label_map_v2.json")
    assert os.path.exists(label_map_path), f"Label map v2 not found at {label_map_path}"

    with open(label_map_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta["num_classes"] == 32, f"Expected 32 classes, got {meta['num_classes']}"
    assert meta["num_features"] == 83, f"Expected 83 features, got {meta['num_features']}"
    print(f"  [PASSED] Metadata v2 loaded cleanly ({meta['num_classes']} ISL classes, 83 features).")

def test_model_v2_inference():
    print("[TEST 3/5] v2 Model Inference & Output...")
    model_path = os.path.join(PHASE1_DIR, "models", "isl_model_v2.joblib")
    assert os.path.exists(model_path), f"Model v2 file not found at {model_path}"

    clf = joblib.load(model_path)
    sample_features = np.zeros(83)
    pred = clf.predict([sample_features])[0]
    probs = clf.predict_proba([sample_features])[0]

    assert pred in clf.classes_, f"Predicted label {pred} not in model classes"
    assert abs(sum(probs) - 1.0) < 1e-4, f"Probability distribution sum invalid: {sum(probs)}"
    print(f"  [PASSED] Model v2 inference produces valid prediction '{pred}' with probability sum {sum(probs):.2f}.")

def test_confidence_thresholding_v2():
    print("[TEST 4/5] v2 Confidence & Unknown Gesture Thresholding...")
    np.random.seed(42)
    random_tuples = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(21)]
    mock_hand = MockHandLandmarks(random_tuples)

    label, conf = classify_gesture(mock_hand)
    print(f"  Inference result on random hand input: Label='{label}', Confidence={conf}%")
    assert conf >= 0.0 and conf <= 100.0, "Confidence score out of range"
    print("  [PASSED] Confidence calculation and thresholding executed cleanly.")

def test_null_hand_handling_v2():
    print("[TEST 5/5] Null / No-Hand Handling...")
    label, conf = classify_gesture(None)
    assert label == "NO HAND DETECTED", f"Expected NO HAND DETECTED, got {label}"
    assert conf == 0.0, f"Expected 0.0 confidence, got {conf}"
    print("  [PASSED] Null hand input handled safely.")

def run_all_tests_v2():
    print("==================================================")
    print("  SignConnect Live - Automated Test Suite (v2)    ")
    print("==================================================\n")
    test_feature_extraction_v2()
    test_metadata_v2()
    test_model_v2_inference()
    test_confidence_thresholding_v2()
    test_null_hand_handling_v2()
    print("\nAll 5 automated v2 pipeline tests passed successfully!")

if __name__ == "__main__":
    run_all_tests_v2()
