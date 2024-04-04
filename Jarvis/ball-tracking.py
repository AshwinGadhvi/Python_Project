import cv2
import math
import time

def distance(x1,y1,x2,y2):
    return math.sqrt((x2-x1)**2+(y2-y1)**2)

def calculate_speed(distance,time_diff):
    return distance/time_diff

previous_time=0
privious_x,privious_y=0,0

cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    if not ret:
        break
    gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    x,y=320,240
    current_time = time.time()
    time_diff = current_time-previous_time
    ball_distance=distance(privious_x,privious_y,x,y)
    speed=calculate_speed(ball_distance,time_diff)
    print(f"Ball speed:{speed} pixels per second")
    previous_time=current_time
    privious_x,privious_y=x,y
    cv2.imshow('Frame',frame)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
cap.released()
cv2.destroyAllWindows()