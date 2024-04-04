import cv2
from cvzone.HandTrackingModule import HandDetector
from playsound import playsound
import time

# Create a video panel
cap = cv2.VideoCapture(0)
# Setting up a size
cap.set(3, 1280)
cap.set(4, 720)

# Checking detection and set up hands
detector = HandDetector(detectionCon=0.8)

# Initialize last played time
last_played_time = time.time()

while True:
    # Give a name to panel
    success, img = cap.read()

    # Finding hands
    hands, img = detector.findHands(img)

    if len(hands) == 1:
        if detector.fingersUp(hands[0]) == [1, 1, 1, 1, 1]:
            if time.time() - last_played_time > 2:  # Check if enough time has passed since last sound played
                playsound('audio/fire.wav')
                last_played_time = time.time()  # Update last played time
        elif detector.fingersUp(hands[0]) == [0, 0, 0, 0, 0]:
            if time.time() - last_played_time > 2:  # Check if enough time has passed since last sound played
                playsound('audio/ironman_chest_rt.wav')
                last_played_time = time.time()  # Update last played time
    elif len(hands) == 2:
        if detector.fingersUp(hands[0]) == [1, 1, 1, 1, 1] and detector.fingersUp(hands[1]) == [1, 1, 1, 1, 1]:
            if time.time() - last_played_time > 2:  # Check if enough time has passed since last sound played
                playsound('audio/iron_man_repulsor.wav')
                last_played_time = time.time()  # Update last played time

    cv2.imshow("Image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
