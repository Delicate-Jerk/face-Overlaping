#1st image overlapping 2nd image working fine

from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import os

app = Flask(__name__)

def detect_face_eyes_nose_mouth(image_data):
    # Your existing code for face, eyes, nose, and mouth detection...
    # Convert image data to OpenCV format
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

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

    # Convert the image to grayscale for face and feature detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None

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
    # Your existing code to save the processed image...
    # Use absolute file path instead of relative file path
    abs_save_path = os.path.join('/Users/arshitarora/Antier-Sol/face-detection', save_path)

    cv2.imwrite(abs_save_path, image)
    
def overlap_images(processed_image_path, overlay_image_path, save_path):
    # Load the processed image with detected facial features
    processed_image = cv2.imread(processed_image_path)

    # Load the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Convert both images to grayscale for face and feature detection
    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
    overlay_gray = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained classifiers for face, eyes, nose, and mouth detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    nose_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_nose.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

    # Detect faces in the processed image
    faces_processed = face_cascade.detectMultiScale(processed_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces_processed) == 0:
        return

    # Detect faces in the overlay image
    faces_overlay = face_cascade.detectMultiScale(overlay_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces_overlay) == 0:
        return

    # Assuming only one face is detected in each image, you can modify this to handle multiple faces
    x, y, w, h = faces_processed[0]
    x_ov, y_ov, w_ov, h_ov = faces_overlay[0]

    # Extract the regions of interest (ROIs) for eyes, nose, and mouth from both images
    eyes_roi_processed = processed_image[y:y + h, x:x + w]
    eyes_roi_overlay = overlay_image[y_ov:y_ov + h_ov, x_ov:x_ov + w_ov]

    # Resize the ROIs from the processed image to match the size of the ROIs in the overlay image
    eyes_roi_processed = cv2.resize(eyes_roi_processed, (w_ov, h_ov))

    # Replace the ROIs in the overlay image with the corresponding ROIs from the processed image
    overlay_image[y_ov:y_ov + h_ov, x_ov:x_ov + w_ov] = eyes_roi_processed

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
