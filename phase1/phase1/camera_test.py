import cv2
import mediapipe as mp

def main():
    # Step 1: Initialize MediaPipe Hands and drawing utilities
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    
    # Configure MediaPipe Hands to detect up to 2 hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Step 2: Open the default webcam (camera index 0)
    cap = cv2.VideoCapture(0)

    # Step 12: Handle webcam initialization failure
    if not cap.isOpened():
        print("Error: Could not open webcam. Please verify your camera is connected and not in use by another application.")
        return

    print("SignConnect - Hand Tracking started successfully!")
    print("Press 'Q' in the camera window to exit.")

    try:
        # Step 3: Read webcam frames continuously
        while cap.isOpened():
            ret, frame = cap.read()

            # Step 13: Handle frame-reading failure safely
            if not ret:
                print("Error: Failed to read frame from webcam.")
                break

            # Step 4: Flip the frame horizontally to create a natural mirror effect
            frame = cv2.flip(frame, 1)

            # Step 5: Convert OpenCV's BGR color format to RGB format required by MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 6: Process the frame and detect hand landmarks
            results = hands.process(rgb_frame)

            # Step 7: Draw 21 hand landmarks and connections if hands are detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

            # Step 8: Display the processed frame in a GUI window
            cv2.imshow("SignConnect - Hand Tracking", frame)

            # Step 9: Press 'Q' or 'q' to exit the application
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("Exiting application...")
                break

    finally:
        # Step 10 & 11: Release webcam and destroy OpenCV windows safely
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Resources released. Application closed.")

if __name__ == "__main__":
    main()
