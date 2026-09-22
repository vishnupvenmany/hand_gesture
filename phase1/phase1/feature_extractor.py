"""
Feature Extractor Module for SignConnect Live (Phase 1)
-------------------------------------------------------
Extracts an 83-dimensional feature vector from MediaPipe hand landmarks:
- 63 scale-normalized wrist-relative 3D coordinates
- 10 pairwise tip-to-tip distances between (Thumb, Index, Middle, Ring, Pinky)
- 5 fingertip-to-wrist distances
- 5 finger extension ratios (tip-to-wrist distance / PIP-to-wrist distance)

Ensures 100% identical feature representation between training data and webcam inference.
"""

import numpy as np

def extract_features(hand_landmarks):
    """
    Extracts an 83-dimensional scale-normalized feature vector from MediaPipe hand landmarks.
    
    Args:
        hand_landmarks: MediaPipe NormalizedLandmarkList (21 landmarks)
        
    Returns:
        list of float: Flattened list of 83 normalized features
    """
    if not hand_landmarks:
        return []

    if hasattr(hand_landmarks, 'landmark') and hand_landmarks.landmark:
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    elif hasattr(hand_landmarks, 'iter_landmarks'):
        pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.iter_landmarks()])
    else:
        return []

    return extract_features_from_numpy(pts)


def extract_features_from_numpy(pts):
    """
    Computes 83 features from a numpy array of shape (21, 3).
    """
    if pts is None or len(pts) != 21:
        return []

    wrist = pts[0]
    diffs = pts - wrist

    # Scale factor: max absolute coordinate displacement from wrist
    scale = np.max(np.abs(diffs))
    if scale < 1e-6:
        scale = 1.0

    # 1. 63 scale-normalized wrist-relative 3D coordinates
    norm_coords = (diffs / scale).flatten()

    tips = [4, 8, 12, 16, 20]
    pips = [3, 6, 10, 14, 18]

    # 2. Pairwise tip-to-tip distances (10 features)
    tip_dists = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            d = np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / scale
            tip_dists.append(d)

    # 3. Fingertip to wrist distances (5 features)
    wrist_dists = [np.linalg.norm(pts[t] - wrist) / scale for t in tips]

    # 4. Finger extension ratios (5 features)
    ext_ratios = []
    for t, p in zip(tips, pips):
        d_t = np.linalg.norm(pts[t] - wrist)
        d_p = np.linalg.norm(pts[p] - wrist) + 1e-6
        ext_ratios.append(d_t / d_p)

    # Combine into 83-dimensional feature vector
    features = np.concatenate([norm_coords, tip_dists, wrist_dists, ext_ratios])
    return features.tolist()