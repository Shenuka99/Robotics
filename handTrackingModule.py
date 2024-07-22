import cv2
import mediapipe as mp
import time
import random

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionConf=0.5, trackingConf=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackingConf = trackingConf

        self.mpHands = mp.solutions.hands
        self.mpdraw = mp.solutions.drawing_utils
        self.hands = self.mpHands.Hands(static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionConf,
            min_tracking_confidence=self.trackingConf)

    def findHands(self, frame, draw=True):
        frame = cv2.flip(frame, 1)
        frame = cv2.rotate(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        self.result = self.hands.process(frame_rgb)

        if self.result.multi_hand_landmarks:
            for hand_landmarks in self.result.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

        return frame
    def findPosition(self, frame, handNo = 0, draw=True):
        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[handNo]


            for index, landmark in enumerate(myHand.landmark):
                # Get the positions of the landmarks
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])
                lmList.append([index, x, y])

                if draw:
                    cv2.circle(frame, (x, y), 7, (255, 0, 0), cv2.WARP_FILL_OUTLIERS)

        return lmList
def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = detector.findHands(frame)
        lmList = detector.findPosition(frame)

        if len(lmList) != 0:
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(frame, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 4)

        cv2.imshow('Window', frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
