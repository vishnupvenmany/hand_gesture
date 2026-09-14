# SignConnect Live

**SignConnect Live** is a B.Tech IT multimedia project designed to bridge communication gaps using sign language recognition and translation technology.

## Project Roadmap (3 Phases)

* **Phase 1: Hand Sign → Text**
  * Step 1: Webcam Hand Tracking *(Current Step)*
  * Future Steps: Dataset collection, feature extraction, model training, sign-to-text recognition.
* **Phase 2: Text/Speech → Sign**
* **Phase 3: Live Video Call with Translation**

---

## Phase 1 - Step 1: Webcam Hand Tracking

This milestone verifies webcam capture and MediaPipe 21-landmark hand tracking. It captures real-time video, flips the display for a mirror effect, processes RGB frames using MediaPipe Hands, detects up to 2 hands, and overlays landmark keypoints and skeleton connections.

---

## Setup & Installation Instructions (Windows)

### 1. Create a Python Virtual Environment

Open PowerShell or Command Prompt in the project folder and run:

```bash
python -m venv venv
```

### 2. Activate the Virtual Environment

On Windows:

```cmd
venv\Scripts\activate
```

*(If using PowerShell, ensure script execution policy allows virtual environment activation, e.g., `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

### 3. Install Dependencies

Install required libraries (`opencv-python` and `mediapipe`):

```bash
pip install -r requirements.txt
```

---

## Running the Application

Run the camera test script:

```bash
python phase1/camera_test.py
```

### Quitting the Application

* Click on the camera window named **`SignConnect - Hand Tracking`**.
* Press the **`Q`** key on your keyboard to safely exit the application and release camera resources.
