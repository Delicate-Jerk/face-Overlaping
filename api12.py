from flask import Flask, request, jsonify
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

    # Convert the image to grayscale for face detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return None, None

    # Assuming only one face is detected in the image, you can modify this to handle multiple faces
    x, y, w, h = faces[0]

    # Extract the region of interest (ROI) for the detected face
    face_roi = img_np[y:y + h, x:x + w]

    return img_np, face_roi


def save_image_with_detection(image, save_path):

    # Use absolute file path instead of relative file path
    abs_save_path = os.path.join('/Users/arshitarora/Antier-Sol/face-detection', save_path)
    cv2.imwrite(abs_save_path, image)


def overlap_images(processed_image_path, overlay_image_path, save_path):
    # Load the processed image with detected facial features
    processed_image = cv2.imread(processed_image_path)

    # Load the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Ensure both images have the same dimensions by resizing the overlay image to match the processed image
    overlay_image = cv2.resize(overlay_image, (processed_image.shape[1], processed_image.shape[0]))

    # Convert both images to grayscale for face and feature detection
    processed_gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained classifiers for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the processed image
    faces_processed = face_cascade.detectMultiScale(processed_gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
    if len(faces_processed) == 0:
        return

    # Assuming only one face is detected in each image, you can modify this to handle multiple faces
    x, y, w, h = faces_processed[0]

    # Create a mask to remove the background of the processed image
    mask = np.zeros(processed_image.shape[:2], dtype=np.uint8)
    mask[y:y + h, x:x + w] = 255

    # Apply the mask to remove the background
    processed_face_roi = cv2.bitwise_and(processed_image, processed_image, mask=mask)

    # Replace the face region in the overlay image with the cropped processed face
    overlay_image[y:y + h, x:x + w] = processed_face_roi[y:y + h, x:x + w]

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

            processed_image, detected_face_roi = detect_face_eyes_nose_mouth(image_data)

            if processed_image is None:
                return jsonify({'error': 'Face not detected'})

            # If face is detected, save the processed image with detected face
            processed_image_path = 'processed_image_with_detections.png'
            save_image_with_detection(processed_image, processed_image_path)

            # Replace this with the actual file path of your overlay image
            overlay_image_path = '/Users/arshitarora/Antier-Sol/face-detection/std.jpg'

            # Overlap the processed_image onto the overlapped_image
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
