import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

video_capture = cv2.VideoCapture(0)

ashish_image = face_recognition.load_image_file("photos/Ashish.jpg")
ashish_encoding = face_recognition.face_encodings(ashish_image)[0]

barbosa_image = face_recognition.load_image_file("photos/Barbosa.jpg")
barbosa_encoding = face_recognition.face_encodings(barbosa_image)[0]

Darshan_image = face_recognition.load_image_file("photos/Darshan.jpg")
Darshan_encoding = face_recognition.face_encodings(Darshan_image)[0]

ronaldo_image = face_recognition.load_image_file("photos/Ronaldo.jpg")
ronaldo_encoding = face_recognition.face_encodings(ronaldo_image)[0]

ashwin_image = face_recognition.load_image_file("photos/ashwin_gadhvi.jpeg")
ashwin_encoding = face_recognition.face_encodings(ashwin_image)[0]

dhruv_image = face_recognition.load_image_file("photos/Dhruv.jpg")
dhruv_encoding = face_recognition.face_encodings(dhruv_image)[0]

hem_image = face_recognition.load_image_file("photos/Hem.jpg")
hem_encoding = face_recognition.face_encodings(hem_image)[0]

sneh_image = face_recognition.load_image_file("photos/Sneh.jpg")
sneh_encoding = face_recognition.face_encodings(sneh_image)[0]

Vrajesh_image = face_recognition.load_image_file("photos/Vrajesh.jpg")
Vrajesh_encoding = face_recognition.face_encodings(Vrajesh_image)[0]

Bhargav_image = face_recognition.load_image_file("photos/Bhargav.jpg")
Bhargav_encoding = face_recognition.face_encodings(Bhargav_image)[0]

# Het_image = face_recognition.load_image_file("photos/Het.jpg")
# Het_encoding = face_recognition.face_encodings(Het_image)[0]



known_face_encoding =[
    ashish_encoding,
    barbosa_encoding,
    Darshan_encoding,
    ronaldo_encoding,
    ashwin_encoding,
    dhruv_encoding,
    hem_encoding,
    sneh_encoding,
    Vrajesh_encoding,
    Bhargav_encoding,
#     Het_encoding,
]

known_face_names=[
    "Ashish",
    "Barbosa",
    "Darshan",
    "Ronaldo",
    "Ashwin",
    "Dhruv",
    "Hem",
    "Sneh",
    "Vrajesh",
    "Bhargav",

]

students = known_face_names.copy()

face_locations = []
face_encodings= []
face_names = []
s=True

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f= open(current_date+'.csv','w+',newline='')
lnwriter = csv.writer(f)

while True:
    _,frame=video_capture.read()
    small_frame = cv2.resize(frame,(0,0),fx=0.25,fy=0.25)
    rgb_small_frame = small_frame[:,:,::-1]
    if s:
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame,face_locations)
        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encoding,face_encoding)
            name=""
            face_distance = face_recognition.face_distance(known_face_encoding,face_encoding)
            best_match_index = np.argmin(face_distance)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
            
            face_names.append(name)
            if name in known_face_names:
                if name in students:
                    students.remove(name)
                    print(students)
                    current_time = now.strftime("%H-%M-%S")
                    lnwriter.writerow([name,current_time])
    cv2.imshow("Attendance System",frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()