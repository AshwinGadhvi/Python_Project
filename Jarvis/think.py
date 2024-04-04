import cv2

def detect_objects():
    # Load pre-trained object detection model
    model = cv2.dnn.readNetFromTensorflow('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco.pbtxt')

    # Create video capture object for accessing the webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame for faster processing
        resized_frame = cv2.resize(frame, (300, 300))

        # Preprocess the frame (convert BGR to RGB, normalize, and resize)
        blob = cv2.dnn.blobFromImage(resized_frame, 0.007843, (300, 300), 127.5)

        # Set the blob as input to the model
        model.setInput(blob)

        # Perform object detection
        detections = model.forward()

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.6:  # Minimum confidence threshold
                class_id = int(detections[0, 0, i, 1])

                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
                start_x, start_y, end_x, end_y = box.astype(int)

                # Draw the bounding box and label
                cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)
                label = f"{class_id}: {confidence:.2f}"
                cv2.putText(frame, label, (start_x, start_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show the frame with detections
        cv2.imshow('Object Detection', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_objects()
