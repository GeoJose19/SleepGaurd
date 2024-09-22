import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

# Initialize the mixer module to play alarm sound
mixer.init()
sound = mixer.Sound('alarm.wav')  # Load alarm sound

# Load the pre-trained haarcascade files for face, left eye, and right eye detection
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files/haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files/haarcascade_righteye_2splits.xml')

# Labels for the prediction output (0 = Closed, 1 = Open)
lbl = ['Close', 'Open']

# Load the trained CNN model
model = load_model('models/cnnCat2.h5')

# Get current working directory
path = os.getcwd()

# Initialize webcam for video capture
cap = cv2.VideoCapture(0)

# Font settings for text on screen
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Variables for tracking score and thickness of alarm rectangle
count = 0
score = 0
thicc = 2

# Start the video frame capture loop
while True:
    ret, frame = cap.read()  # Read each frame from the webcam
    height, width = frame.shape[:2]  # Get the frame's dimensions

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale for processing

    # Detect faces and eyes in the frame
    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    # Draw a black rectangle at the bottom of the frame for text display
    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    # Loop through detected faces and draw rectangles around them
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    # Variables to hold prediction values for each eye
    rpred = [99]
    lpred = [99]

    # Process right eye if detected
    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]  # Crop the eye region
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)  # Convert it to grayscale
        r_eye = cv2.resize(r_eye, (24, 24))  # Resize to 24x24 as required by the model
        r_eye = r_eye / 255  # Normalize pixel values
        r_eye = r_eye.reshape(24, 24, -1)  # Reshape to match the input shape of the model
        r_eye = np.expand_dims(r_eye, axis=0)
        rpred = model.predict(r_eye)  # Predict using the CNN model
        rpred = np.argmax(rpred, axis=1)  # Get the class with the highest score (0 or 1)
        break  # Only process the first detected right eye

    # Process left eye if detected (similar to right eye)
    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)
        lpred = model.predict(l_eye)
        lpred = np.argmax(lpred, axis=1)
        break

    # Check if both eyes are closed
    if rpred[0] == 0 and lpred[0] == 0:
        score += 1  # Increase the score if both eyes are closed
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        score -= 1  # Decrease the score if eyes are open
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # Ensure score does not go below 0
    if score < 0:
        score = 0

    # Display score on the screen
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    # If score crosses 15 (drowsiness detected), play the alarm
    if score > 15:
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)  # Save an image of the frame
        try:
            sound.play()  # Play the alarm sound
        except:
            pass

        # Dynamically adjust rectangle thickness for visual alert
        if thicc < 16:
            thicc += 2
        else:
            thicc -= 2
            if thicc < 2:
                thicc = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)  # Draw alert rectangle

    # Display the frame with detections and alerts
    cv2.imshow('frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows when done
cap.release()
cv2.destroyAllWindows()
