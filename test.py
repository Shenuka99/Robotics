import cv2
import mediapipe as mp

# Initialize MediaPipe hands module and drawing utilities
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Set up the video capture
cap = cv2.VideoCapture(0)

fingertips_ids = [8, 12, 16, 20]
pip_joints_ids = [5, 9, 13, 17]

thumbs_up_right = False
thumbs_up_left = False
closed = False
opened = False

def check_hand_sign(hand_landmarks,label, hand_side):
    global thumbs_up_right, thumbs_up_left, opened, closed

    # Checking for closed hands, both right and left for front and backsides
    if ((hand_landmarks.landmark[8].y > hand_landmarks.landmark[5].y)
        and (hand_landmarks.landmark[12].y > hand_landmarks.landmark[9].y)
        and (hand_landmarks.landmark[16].y > hand_landmarks.landmark[13].y)
        and (hand_landmarks.landmark[20].y > hand_landmarks.landmark[17].y)
        and (hand_landmarks.landmark[17].y) < (hand_landmarks.landmark[1].y)
        and (
                ( hand_side == 'front' and
                (
                    (label == 'Right' and (hand_landmarks.landmark[3].x < hand_landmarks.landmark[4].x))
                    or (label == 'Left' and (hand_landmarks.landmark[3].x > hand_landmarks.landmark[4].x))
                )
                )
            or
                (hand_side == 'back' and
                (
                    (label == 'Right' and (hand_landmarks.landmark[3].x > hand_landmarks.landmark[4].x))
                    or (label == 'Left' and (hand_landmarks.landmark[3].x < hand_landmarks.landmark[4].x))
                )
                )
            )
        ):

        closed = True
        opened = False
        thumbs_up_right = False
        thumbs_up_left = False
    # else:
    #     closed = False
    #     opened = False
    #     thumbs_up_right = False
    #     thumbs_up_left = False

    # Checking for opened hand for both right and left
    elif (
            (hand_landmarks.landmark[8].y < hand_landmarks.landmark[5].y)
        and (hand_landmarks.landmark[12].y < hand_landmarks.landmark[9].y)
        and (hand_landmarks.landmark[16].y < hand_landmarks.landmark[13].y)
        and (hand_landmarks.landmark[20].y < hand_landmarks.landmark[17].y)
        and (hand_landmarks.landmark[17].y) < (hand_landmarks.landmark[1].y)
        and (
            (hand_side == 'front' and
             (
                     (label == 'Right' and (hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x))
                     or (label == 'Left' and (hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x))
             )
            )
            or
            (hand_side == 'back' and
             (
                     (label == 'Right' and (hand_landmarks.landmark[4].x > hand_landmarks.landmark[2].x))
                     or (label == 'Left' and (hand_landmarks.landmark[4].x < hand_landmarks.landmark[2].x))
             )
            )
    )

        ) :
        closed = False
        opened = True
        thumbs_up_right = False
        thumbs_up_left = False

    elif (hand_landmarks.landmark[17].y) > (hand_landmarks.landmark[0].y):
            closed = False
            opened = False
            if label == 'Right':
                print('HERE')
                if ((hand_landmarks.landmark[8].x > hand_landmarks.landmark[6].x)
                        and (hand_landmarks.landmark[12].x > hand_landmarks.landmark[10].x)
                        and (hand_landmarks.landmark[16].x > hand_landmarks.landmark[14].x)
                        and (hand_landmarks.landmark[20].x > hand_landmarks.landmark[18].x)
                        and (hand_landmarks.landmark[5].y > hand_landmarks.landmark[4].y)):
                    thumbs_up_right = True
                else:
                    thumbs_up_right = False
                    # print(thumbs_up_right)

            if label == 'Left':
                if ((hand_landmarks.landmark[8].x < hand_landmarks.landmark[6].x)
                        and (hand_landmarks.landmark[12].x < hand_landmarks.landmark[10].x)
                        and (hand_landmarks.landmark[16].x < hand_landmarks.landmark[14].x)
                        and (hand_landmarks.landmark[20].x < hand_landmarks.landmark[18].x)
                        and (hand_landmarks.landmark[5].y > hand_landmarks.landmark[4].y)):
                    thumbs_up_left = True
                else:
                    thumbs_up_left = False


def which_hand_side(hand_landmarks, label):
    # Get the y-coordinates of the index and pinky finger tips
    index_finger_tip = hand_landmarks.landmark[8].x
    pinky_finger_tip = hand_landmarks.landmark[20].x

    hand_side = 'front'

    if label == 'Right':
        if index_finger_tip > pinky_finger_tip:
            hand_side = 'back'
    elif label == 'Left':
        if index_finger_tip < pinky_finger_tip:
            hand_side = 'back'

    return hand_side



# def is_thumbs_up(hand_landmarks, label):
#     global thumbs_up_right, thumbs_up_left


# Initialize the hands object with the required parameters
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.7) as hands:
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
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                label = handedness.classification[0].label
                score = handedness.classification[0].score
                hand_side = which_hand_side(hand_landmarks, label)
                check_hand_sign(hand_landmarks, label, hand_side)
                # is_thumbs_up(hand_landmarks, label)

                # # Check if the hand is closed

                if label == "Left":
                    if closed:
                        cv2.putText(frame, "Left Hand Closed", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if opened:
                        cv2.putText(frame, "Left Hand Open", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f'Hand side: {hand_side}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,cv2.LINE_AA)
                    if thumbs_up_left:
                        cv2.putText(frame, "Thumbs up left", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                if label == 'Right':
                    if closed:
                        cv2.putText(frame, "Right Hand Closed", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    if opened:
                        cv2.putText(frame, "Right Hand Open", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.putText(frame, f'Hand side: {hand_side}', (300, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,cv2.LINE_AA)
                    if thumbs_up_right:
                        cv2.putText(frame, "Thumbs up right", (300, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


        # Display the frame
        cv2.imshow('Hand Pose Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
