import cv2
import mediapipe as mp

# Initialize MediaPipe hands module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the video capture
cap = cv2.VideoCapture(0)


def is_hand_closed(hand_landmarks):
    # Define the landmarks for finger tips and PIP joints
    fingertips_ids = [ 8, 12, 16, 20]
    pip_joints_ids = [ 6, 10, 14, 18]

    print(hand_landmarks)

    closed = True
    for tip_id, pip_id in zip(fingertips_ids, pip_joints_ids):
        # Check if fingertip is below the PIP joint (folded finger)
        if hand_landmarks.landmark[tip_id].y < hand_landmarks.landmark[pip_id].y:
            closed = False
            break
    # print(closed)
    return closed


# Initialize the hands object with the required parameters
with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.7) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a later selfie-view display
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Check if the hand is closed
                if is_hand_closed(hand_landmarks):
                    cv2.putText(frame, "Hand Closed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "Hand Open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the frame
        cv2.imshow('Hand Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
