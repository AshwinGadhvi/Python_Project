import cv2
from cvzone.HandTrackingModule import HandDetector

#create a video panel
cap = cv2.VideoCapture(0)
#setting up a size
cap.set(3,1280)
cap.set(4,720)

#checking detection and set up hands
detector = HandDetector(detectionCon=0.8)
startDistance = None
scale = 0
cx,cy=500,500


#infinite loop
while True:
    #give a name to panel
    success, img = cap.read()

    #finding hands
    hands,img = detector.findHands(img)
    
    #posting image on panel
    img1 = cv2.imread("image/ashwin.jpg")

    if len(hands)==2:
        #print hands fingers
        # print(detector.fingersUp(hands[0]),detector.fingersUp(hands[1]))
        if detector.fingersUp(hands[0]) == [1,1,0,0,0] and detector.fingersUp(hands[1]) == [1,1,0,0,0]:
            # print("Zoom Gesture")
            lmList1 = hands[0]["lmList"]
            lmList2 = hands[1]["lmList"]
            if startDistance is None :
            #point 8 is the tip of finger
            #error too many values to unpack expected 2
                #length, info, img= detector.findDistance(lmList1[8],lmList2[8],img)  
                length, info, img= detector.findDistance(hands[0]["center"],hands[1]["center"],img)  
                print(length)
                startDistance = length
                #length, info, img= detector.findDistance(lmList1[8],lmList2[8],img) 
            length, info, img= detector.findDistance(hands[0]["center"],hands[1]["center"],img)  
            scale = int((length - startDistance)//2) 
            cx,cy=info[4:]
            print(scale)
    else:
        startDistance = None
    try:
        h1,w1,_=img1.shape
        newH,newW = ((h1+scale)//2)*2,((w1+scale)//2)*2
        img1 = cv2.resize(img1,(newW,newH))
        img[cy-newH//2:cy+newH//2,cx-newW//2:cx+newW//2]= img1
    except:
        pass

        
    cv2.imshow("Image",img)
    cv2.waitKey(1)