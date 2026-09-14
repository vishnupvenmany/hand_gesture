def extract_features(hand_landmarks):
    wrist = hand_landmarks.landmark[0]

    features = []

    for landmark in hand_landmarks.landmark:
        x = landmark.x - wrist.x
        y = landmark.y - wrist.y
        z = landmark.z - wrist.z

        features.extend([x, y, z])

    return features