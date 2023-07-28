#working perfectly with all feature detection but with no exception if no face detected

from flask import Flask, request, jsonify
from PIL import Image
import io
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

def detect_face_eyes_nose_mouth(image_data):
    # Convert image data to OpenCV format
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Load a pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load a pre-trained eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load a pre-trained nose detector
    nose_cascade_path = '/Users/arshitarora/Antier-Sol/face-detection/haarcascade_mcs_nose.xml'
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

    # Load a pre-trained mouth detector
    mouth_cascade_path = '/Users/arshitarora/Antier-Sol/face-detection/haarcascade_mcs_mouth.xml'
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

    # Convert the image to grayscale for face and feature detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(img_np, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Detect eyes in the face region
        roi_gray = gray[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(img_np, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)

        # Detect nose in the face region
        nose = nose_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
        for (nx, ny, nw, nh) in nose:
            cv2.rectangle(img_np, (x + nx, y + ny), (x + nx + nw, y + ny + nh), (0, 0, 255), 2)

        # Detect mouth in the face region
        mouth = mouth_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
        for (mx, my, mw, mh) in mouth:
            cv2.rectangle(img_np, (x + mx, y + my), (x + mx + mw, y + my + mh), (255, 0, 255), 2)

    return img_np

def save_image_with_detection(image, save_path):
    # Use absolute file path instead of relative file path
    abs_save_path = os.path.join('/Users/arshitarora/Antier-Sol/face-detection', save_path)

    cv2.imwrite(abs_save_path, image)

@app.route('/process_image', methods=['POST'])
def upload_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        if file:
            image_data = file.read()

            processed_image = detect_face_eyes_nose_mouth(image_data)

            # If face, eyes, nose, and mouth are detected, send the processed image and save it to disk
            _, buffer = cv2.imencode('.png', processed_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Save the processed image with detected face, eyes, nose, and mouth
            save_image_with_detection(processed_image, 'processed_image_with_detections.png')

            return jsonify({'message': 'Face, eyes, nose, and mouth detected', 'processed_image': encoded_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
