import cv2
import face_recognition
from deepface import DeepFace # type: ignore

def detect_age_and_emotion(frame):
    # Convert the frame to RGB format for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find faces in the frame
    face_locations = face_recognition.face_locations(rgb_frame)

    # If no face is detected, return None
    if not face_locations:
        return None, None

    # Extract face landmarks
    face_landmarks_list = face_recognition.face_landmarks(rgb_frame)

    # Calculate the average age
    ages = [face_recognition.face_distance([face_recognition.face_encodings(rgb_frame, [face_location])[0]],
                                           face_recognition.api.face_encodings([face_recognition.api.load_image_file(rgb_frame)])[0])[0]
            for face_location in face_locations]
    average_age = sum(ages) / len(ages)

    # Emotion detection
    emotion_labels = []
    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = rgb_frame[top:bottom, left:right]
        result = DeepFace.analyze(face_image, actions=['emotion'])
        emotion_label = result['dominant_emotion']
        emotion_labels.append(emotion_label)

    return average_age, emotion_labels

def detect_faces(video_source=0):
    # Open a connection to the camera (use 0 for the default camera)
    cap = cv2.VideoCapture(video_source)

    # Load the pre-trained face detection classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        # Draw rectangles around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Age and emotion detection
        average_age, emotion_labels = detect_age_and_emotion(frame)

        # Display age and emotion information
        if average_age is not None:
            cv2.putText(frame, f"Average Age: {average_age:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
        if emotion_labels:
            cv2.putText(frame, f"Emotion: {', '.join(emotion_labels)}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Face Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Call the function with the default camera (0). Change to a different value if using an external camera.
    detect_faces()
