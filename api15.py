#1st image overlapping 2nd image working fine

from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import os
import face_recognition

app = Flask(__name__)

def detect_face_eyes_nose_mouth(image_data):
    # Convert image data to OpenCV format
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Use face_recognition library for face detection and facial landmarks
    face_locations = face_recognition.face_locations(img_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(img_np, face_locations)

    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):
        # Draw a rectangle around the face
        cv2.rectangle(img_np, (left, top), (right, bottom), (255, 0, 0), 2)

        # Extract the face region for further feature detection
        face_roi_gray = cv2.cvtColor(img_np[top:bottom, left:right], cv2.COLOR_BGR2GRAY)

            # Load a pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Load a pre-trained eye detector
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

    # Load a pre-trained nose detector
    nose_cascade_path = 'haarcascade_mcs_nose.xml'
    nose_cascade = cv2.CascadeClassifier(nose_cascade_path)

    # Load a pre-trained mouth detector
    mouth_cascade_path = 'haarcascade_mcs_mouth.xml'
    mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

        # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(25, 25))
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img_np, (left + ex, top + ey), (left + ex + ew, top + ey + eh), (0, 255, 0), 2)

    # Detect nose in the face region
    nose = nose_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
    for (nx, ny, nw, nh) in nose:
        cv2.rectangle(img_np, (left + nx, top + ny), (left + nx + nw, top + ny + nh), (0, 0, 255), 2)

    # Detect mouth in the face region
    mouth = mouth_cascade.detectMultiScale(face_roi_gray, scaleFactor=1.8, minNeighbors=20, minSize=(25, 25))
    for (mx, my, mw, mh) in mouth:
        cv2.rectangle(img_np, (left + mx, top + my), (left + mx + mw, top + my + mh), (255, 0, 255), 2)

    return img_np


def save_image_with_detection(image, save_path):
    # Your existing code to save the processed image...
    # Use absolute file path instead of relative file path
    abs_save_path = os.path.join('/Users/arshitarora/Antier-Sol/face-detection', save_path)

    cv2.imwrite(abs_save_path, image)
    
def overlap_images(processed_image_path, overlay_image_path, save_path):
    # Load the processed image with detected facial features
    processed_image = cv2.imread(processed_image_path)

    # Load the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Convert both images to grayscale for face detection
    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    overlay_gray = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)

    # Use face_recognition library for face detection and facial landmarks
    processed_face_locations = face_recognition.face_locations(processed_image, model='hog')
    processed_face_landmarks = face_recognition.face_landmarks(processed_image, processed_face_locations)

    overlay_face_locations = face_recognition.face_locations(overlay_image, model='hog')
    overlay_face_landmarks = face_recognition.face_landmarks(overlay_image, overlay_face_locations)

    if not processed_face_landmarks or not overlay_face_landmarks:
        return

    # Assuming only one face is detected in each image, you can modify this to handle multiple faces
    processed_top, processed_right, processed_bottom, processed_left = processed_face_locations[0]
    overlay_top, overlay_right, overlay_bottom, overlay_left = overlay_face_locations[0]

    # Extract the regions of interest (ROIs) for the faces from both images
    processed_face_roi = processed_image[processed_top:processed_bottom, processed_left:processed_right]
    overlay_face_roi = overlay_image[overlay_top:overlay_bottom, overlay_left:overlay_right]

    # Resize the processed face to match the size of the overlay face
    overlay_face_roi_resized = cv2.resize(processed_face_roi, (overlay_right - overlay_left, overlay_bottom - overlay_top))

    # Replace the face region in the overlay image with the processed face region
    overlay_image[overlay_top:overlay_bottom, overlay_left:overlay_right] = overlay_face_roi_resized

    # Save the updated overlapped image
    cv2.imwrite(save_path, overlay_image)


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

            if processed_image is None:
                return jsonify({'error': 'Face not detected'})

            # If face, eyes, nose, and mouth are detected, send the processed image and save it to disk
            _, buffer = cv2.imencode('.png', processed_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Save the processed image with detected face, eyes, nose, and mouth
            processed_image_path = 'processed_image_with_detections.png'
            save_image_with_detection(processed_image, processed_image_path)

            # Replace this with the actual file path of your overlay image
            overlay_image_path = '/Users/arshitarora/Antier-Sol/face-detection/Unknown.jpg'

            # Overlap the images and save the updated overlapped image
            overlap_save_path = 'overlapped_image.png'
            overlap_images(processed_image_path, overlay_image_path, overlap_save_path)

            # Read the overlapped image and convert it to base64 for sending in the response
            overlapped_image = cv2.imread(overlap_save_path)
            _, buffer = cv2.imencode('.png', overlapped_image)
            encoded_overlapped_image = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'message': 'Image overlapped and saved', 'overlapped_image': encoded_overlapped_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
