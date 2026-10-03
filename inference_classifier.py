"""Module 3: Live ISL -> Text. Press 'q' to quit."""
import cv2

from gesture_recognizer import GestureRecognizer

recognizer = GestureRecognizer()

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Could not read from webcam")
        break

    recognizer.predict(frame)

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
