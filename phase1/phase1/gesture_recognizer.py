"""
ML-based Gesture Recognizer Module for SignConnect Live (Phase 1) - v2 Model
-----------------------------------------------------------------------------
Loads the trained regularized MLP classifier (isl_model_v2.joblib) and predicts Indian Sign Language (ISL)
signs from 83-dimensional scale-normalized MediaPipe hand features with honest probability confidence scoring.
"""

import os
import json
import joblib

try:
    from phase1.feature_extractor import extract_features
except ImportError:
    from feature_extractor import extract_features

# Paths for v2 model
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_model_v2.joblib")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map_v2.json")

# Baseline fallback path if v2 not trained
BASELINE_MODEL_PATH = os.path.join(BASE_DIR, "models", "isl_model.joblib")
BASELINE_LABEL_MAP_PATH = os.path.join(BASE_DIR, "models", "label_map.json")

# Global singleton model loader
_MODEL = None
_METADATA = None

CONFIDENCE_THRESHOLD = 0.40  # Probability threshold for low-confidence fallback


def load_model_if_needed():
    """Loads the trained ML model v2 (or baseline fallback) and metadata from disk."""
    global _MODEL, _METADATA

    if _MODEL is None:
        target_model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else BASELINE_MODEL_PATH
        target_meta_path  = LABEL_MAP_PATH if os.path.exists(LABEL_MAP_PATH) else BASELINE_LABEL_MAP_PATH

        if not os.path.exists(target_model_path):
            raise FileNotFoundError(f"Trained model file not found at: {target_model_path}. Please run train_model_v2.py first.")

        _MODEL = joblib.load(target_model_path)

        if os.path.exists(target_meta_path):
            with open(target_meta_path, 'r', encoding='utf-8') as f:
                _METADATA = json.load(f)
        else:
            _METADATA = {"classes": list(_MODEL.classes_)}


def classify_gesture(hand_landmarks):
    """
    Classifies 21 MediaPipe hand landmarks into a predicted ISL sign with confidence score.
    
    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList (21 points)
        
    Returns:
        tuple (str, float): (predicted_sign, confidence_percentage)
                            e.g. ("WATER", 92.5) or ("LOW CONFIDENCE (S)", 34.1)
    """
    if not hand_landmarks:
        return "NO HAND DETECTED", 0.0

    # 1. Extract 83-dimensional scale-normalized features
    features = extract_features(hand_landmarks)
    if not features or len(features) < 63:
        return "INVALID LANDMARKS", 0.0

    # 2. Ensure model is loaded
    load_model_if_needed()

    # 3. Predict class probabilities
    probabilities = _MODEL.predict_proba([features])[0]
    max_idx = int(probabilities.argmax())
    confidence = float(probabilities[max_idx])
    predicted_label = str(_MODEL.classes_[max_idx])

    # 4. Confidence Thresholding (Prevent false positive claims)
    if confidence < CONFIDENCE_THRESHOLD:
        return f"LOW CONFIDENCE ({predicted_label})", round(confidence * 100, 1)

    return predicted_label, round(confidence * 100, 1)
