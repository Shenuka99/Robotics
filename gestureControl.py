import cv2
import time
import numpy as np
import handTrackingModule as htm
import math
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 1024, 768
cap = cv2.VideoCapture(0)
pTime = 0

# Set the width and height of the frame
cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

# Verify the width and height
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

print(f"Requested resolution: {wCam}x{hCam}")
print(f"Actual resolution: {int(actual_width)}x{int(actual_height)}")

detector = htm.handDetector(detectionConf=0.7)


devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)
volume.GetMute()
volume.GetMasterVolumeLevel()

volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]


while True:
    success, frame = cap.read()
    frame = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)
    if len(lmList) != 0:
        print(lmList)

        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(frame, (x1,y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(frame, (x2,y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
        cv2.circle(frame, (cx,cy), 10, (255, 0, 255), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        volume_level = np.interp(length, [10, 180], [minVol, maxVol])
        print(volume_level)
        volume.SetMasterVolumeLevel(volume_level, None)

        if length < 50:
            cv2.circle(frame, (cx, cy), 10, (0, 255, 0), cv2.FILLED)



    # Resize frame to the desired dimensions, if needed
    frame = cv2.resize(frame, (wCam, hCam))

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(frame, f"FPS : {(int(fps))}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

    cv2.imshow('Window', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
