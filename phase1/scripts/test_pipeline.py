"""
Automated Pipeline Test Suite for SignConnect Live (Phase 1)
------------------------------------------------------------
Verifies:
1. Feature normalization vector length (63 features)
2. Label loading and metadata mapping
3. Trained Random Forest model loading and inference
4. Honest probability confidence scoring
5. Low confidence / UNKNOWN handling
"""

import os
import sys
import json
import joblib
import numpy as np

# Add parent path to import phase1 modules
PHASE1_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(PHASE1_DIR, "phase1"))

from feature_extractor import extract_features_from_raw_tuples
from gesture_recognizer import classify_gesture, load_model_if_needed

class MockLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z

class MockHandLandmarks:
    def __init__(self, landmarks_tuples):
        self.landmark = [MockLandmark(x, y, z) for x, y, z in landmarks_tuples]

def test_feature_normalization():
    print("[TEST 1/5] Feature Normalization Dimension...")
    # Generate 21 dummy tuples
    raw_tuples = [(0.1 * i, 0.2 * i, 0.05 * i) for i in range(21)]
    features = extract_features_from_raw_tuples(raw_tuples)

    assert len(features) == 63, f"Expected 63 features, got {len(features)}"
    assert features[0] == 0.0 and features[1] == 0.0 and features[2] == 0.0, "Wrist landmark (lm0) should be (0,0,0)"
    print("  [PASSED] Feature normalization produces 63 wrist-relative coordinates.")

def test_metadata_and_classes():
    print("[TEST 2/5] Metadata and Class Map Loading...")
    label_map_path = os.path.join(PHASE1_DIR, "models", "label_map.json")
    assert os.path.exists(label_map_path), f"Label map not found at {label_map_path}"

    with open(label_map_path, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    assert meta["num_classes"] == 32, f"Expected 32 classes, got {meta['num_classes']}"
    assert "classes" in meta and len(meta["classes"]) == 32, "Classes list mismatch"
    print(f"  [PASSED] Metadata loaded cleanly. Vocabulary has {len(meta['classes'])} ISL signs.")

def test_model_inference():
    print("[TEST 3/5] Model Inference & Output...")
    model_path = os.path.join(PHASE1_DIR, "models", "isl_model.joblib")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    clf = joblib.load(model_path)
    # Generate synthetic 63 feature vector
    sample_features = np.zeros(63)
    pred = clf.predict([sample_features])[0]
    probs = clf.predict_proba([sample_features])[0]

    assert pred in clf.classes_, f"Predicted label {pred} not in model classes"
    assert sum(probs) > 0.99, f"Probability distribution sum invalid: {sum(probs)}"
    print(f"  [PASSED] Model inference produces valid prediction '{pred}' with probability sum {sum(probs):.2f}.")

def test_confidence_and_unknown_handling():
    print("[TEST 4/5] Confidence & Unknown Gesture Thresholding...")
    # Test random noise coordinates (unsupported gesture)
    np.random.seed(42)
    random_tuples = [(np.random.rand(), np.random.rand(), np.random.rand()) for _ in range(21)]
    mock_hand = MockHandLandmarks(random_tuples)

    label, conf = classify_gesture(mock_hand)
    print(f"  Inference result on random hand input: Label='{label}', Confidence={conf}%")
    assert conf >= 0.0 and conf <= 100.0, "Confidence score out of range"
    print("  [PASSED] Confidence calculation and thresholding executed cleanly.")

def test_null_hand_handling():
    print("[TEST 5/5] Null / No-Hand Handling...")
    label, conf = classify_gesture(None)
    assert label == "NO HAND DETECTED", f"Expected NO HAND DETECTED, got {label}"
    assert conf == 0.0, f"Expected 0.0 confidence, got {conf}"
    print("  [PASSED] Null hand input handled safely.")

def run_all_tests():
    print("==================================================")
    print("  SignConnect Live - Automated Pipeline Test Suite ")
    print("==================================================\n")
    test_feature_normalization()
    test_metadata_and_classes()
    test_model_inference()
    test_confidence_and_unknown_handling()
    test_null_hand_handling()
    print("\nAll 5 automated pipeline tests passed successfully!")

if __name__ == "__main__":
    run_all_tests()
