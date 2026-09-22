# SignConnect Live - Phase 1: Indian Sign Language (ISL) Recognition (v2)

**SignConnect Live** is a CUSAT B.Tech IT multimedia project designed to convert **Indian Sign Language (ISL)** gestures into text in real time using computer vision and machine learning.

---

## 1. Overview & Generalization Summary

Phase 1 features an improved, regularized **ISL Sign → Text** recognition engine. The system was specifically optimized to **reduce overfitting and improve generalization** across different signers, hand sizes, and distances from the camera.

### Baseline vs. Improved (v2) Metrics Comparison

| Metric | Baseline (`isl_model.joblib`) | Improved (`isl_model_v2.joblib`) | Improvement / Delta |
| :--- | :---: | :---: | :---: |
| **Train Accuracy** | 100.00% | **83.58%** | Overfitting reduced by 16.42% |
| **Validation Accuracy** | 67.26% | **70.83%** | **+3.57%** |
| **Held-Out Test Accuracy** | 63.69% | **68.45%** | **+4.76%** |
| **Test Macro F1** | 0.6209 | **0.6636** | **+0.0427** |
| **Test Weighted F1** | 0.6257 | **0.6680** | **+0.0423** |
| **Generalization Gap** | 36.31% | **15.13%** | **Gap Reduced by 21.18%!** |

---

## 2. Feature Extraction Pipeline (83 Features)

The v2 feature extractor transforms 21 MediaPipe 3D hand keypoints into an 83-dimensional scale-normalized representation:

1. **Scale-Normalized Wrist-Relative 3D Coordinates (63 features)**:
   Wrist $(x_0, y_0, z_0)$ is subtracted from all 21 keypoints and scaled by the maximum coordinate displacement:
   $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{p}_i - \mathbf{p}_0}{\max(|\mathbf{p} - \mathbf{p}_0|)}$$
2. **Pairwise Fingertip-to-Fingertip Distances (10 features)**: Pairwise Euclidean distances between Thumb (4), Index (8), Middle (12), Ring (16), and Pinky (20) tips.
3. **Fingertip-to-Wrist Distances (5 features)**: Euclidean distance from wrist (0) to all 5 fingertips.
4. **Finger Extension Ratios (5 features)**: Ratio of fingertip distance to PIP joint distance relative to wrist:
   $$R_k = \frac{\|\mathbf{p}_{\text{tip}} - \mathbf{p}_0\|}{\|\mathbf{p}_{\text{pip}} - \mathbf{p}_0\| + \epsilon}$$

---

## 3. Dataset & Safe Data Augmentation

* **Dataset**: **SignBridge-AI ISL Hand Landmark Corpus**
* **Vocabulary Size**: **32 ISL Classes** (Digits `1`–`5`, Alphabets `A`–`Z`, Word gesture `HELLO`)
* **Sample Count**:
  * **Train Set**: 784 samples (`train.csv`) $\xrightarrow{\text{4x Augmentation}}$ 3,136 samples
  * **Validation Set**: 168 samples (`val.csv`)
  * **Held-Out Test Set**: 168 samples (`test.csv`)
* **Augmentation**: Small, physically plausible transformations applied **strictly to training data**:
  * Minor 2D rotation: $\pm 10^\circ$
  * Minor scaling: $0.95 \dots 1.05$
  * Coordinate Gaussian jitter: $\sigma = 0.005$

---

## 4. Machine Learning Model Architecture

* **Model**: **Regularized Multi-Layer Perceptron (MLP)**
* **Configuration**: `hidden_layer_sizes=(256, 128)`, `alpha=0.10` (L2 Regularization), `early_stopping=True`.
* **Selection Strategy**: Hyperparameter configuration selected strictly using **Validation Accuracy & Validation Macro F1** before single evaluation on the held-out test set.

---

## 5. Execution Commands (PowerShell)

### Download ISL Dataset
```powershell
cd C:\Users\vishn\Desktop\handgesture
.\venv311\Scripts\python.exe phase1/scripts/download_dataset.py
```

### Train & Evaluate Improved Model (v2)
```powershell
.\venv311\Scripts\python.exe phase1/scripts/train_model_v2.py
```

### Run Automated Pipeline Test Suite (v2)
```powershell
.\venv311\Scripts\python.exe phase1/scripts/test_pipeline_v2.py
```

### Launch Real-Time Live ISL Webcam Recognition
```powershell
.\venv311\Scripts\python.exe phase1/phase1/camera_test.py
```
* Press **`Q`** key in camera window to quit.

---

## 6. Project Limitations

1. **Vocabulary Limit**: Recognizes 32 specific ISL classes; signs outside this vocabulary trigger `LOW CONFIDENCE`.
2. **Hand Landmarks Only**: Facial expressions and body posture are not tracked.
3. **Lighting & Occlusion**: Extreme low light or hand occlusion can affect landmark tracking quality.
