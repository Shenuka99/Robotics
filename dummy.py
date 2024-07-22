import cv2
import mediapipe as mp
import time
import handTrackingModule as htm


pTime = 0
cTime = 0
cap = cv2.VideoCapture(0)
detector = htm.handDetector()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame = detector.findHands(frame, draw=False)
    lmList = detector.findPosition(frame, draw=False)

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
