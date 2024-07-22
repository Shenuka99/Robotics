import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize the hands model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open the default webcam
cap = cv2.VideoCapture(0)


# Function to generate a unique color for each landmark
def get_color(index):
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
        (0, 255, 255), (255, 0, 255), (192, 192, 192), (128, 0, 0),
        (128, 128, 0), (0, 128, 0), (128, 0, 128), (0, 128, 128),
        (0, 0, 128), (255, 165, 0), (255, 215, 0), (0, 191, 255),
        (135, 206, 250), (147, 112, 219), (255, 20, 147), (255, 182, 193),
        (240, 230, 140)
    ]
    return colors[index % len(colors)]


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)


    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the RGB image to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for index, landmark in enumerate(hand_landmarks.landmark):
                # Get the positions of the landmarks
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw circles at the landmarks
                color = get_color(index)
                print(color)
                cv2.circle(frame, (x, y), 5, color, -1)

                # Annotate the landmarks
                cv2.putText(frame, f'{index}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Display the resulting frame
    cv2.imshow('Hand Landmarks with IDs and Colors', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
