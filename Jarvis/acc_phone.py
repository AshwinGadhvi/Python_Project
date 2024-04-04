import cv2
url='https://'
cap=cv2.VideoCapture(url)
while(True):
    ret,frame=cap.read()
    if frame is not None:
        cv2.imShow('frame',frame)
    q=cv2.waitKey(1)
    if q == ord("q"):
        break
cv2.destroyAllWindows()