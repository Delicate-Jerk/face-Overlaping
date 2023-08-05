from flask import Flask, request, jsonify
import cv2
import base64
import numpy as np
import os
import face_recognition

app = Flask(__name__)

def draw_ellipse_around_points(image, points):
    for point in points:
        center = tuple(point)
        axes_length = (15, 25)  # You can adjust the axes length to change the shape of the ellipse
        angle = 0  # The rotation angle of the ellipse (0 means no rotation)
        start_angle = 0  # The starting angle of the elliptical arc
        end_angle = 360  # The ending angle of the elliptical arc
        color = (0, 255, 0)  # Color of the ellipse (green in this case)
        thickness = -1  # -1 means the ellipse will be filled

        cv2.ellipse(image, center, axes_length, angle, start_angle, end_angle, color, thickness)


def detect_face_eyes_nose_mouth(image_data):
    # ... (existing code for face detection and feature extraction remains the same)
    img_np = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)

    # Use face_recognition library for face detection and facial landmarks
    face_locations = face_recognition.face_locations(img_np, model='hog')
    face_landmarks = face_recognition.face_landmarks(img_np, face_locations)

    for (top, right, bottom, left), landmarks in zip(face_locations, face_landmarks):

        # Create a mask with a black image of the same size as the original image
        mask = np.zeros_like(img_np)

        # Draw an ellipse on the mask (white color) using the face bounding box as the parameters
        center = ((left + right) // 2, (top + bottom) // 2)
        axes_length = ((right - left) // 2, (bottom - top) // 2)
        cv2.ellipse(mask, center, axes_length, angle=0, startAngle=0, endAngle=360, color=(255, 255, 255), thickness=-4)

        # Use the mask to keep only the elliptical region from the original image
        img_np = cv2.bitwise_and(img_np, mask)

    return img_np

def wrap_and_blend_images(processed_image_path, overlay_image_path, background_image_path, save_path, alpha=0.5):
    # Load the processed image with detected facial features
    processed_image = cv2.imread(processed_image_path)

    # Load the overlay image
    overlay_image = cv2.imread(overlay_image_path)

    # Load the background image
    background_image = cv2.imread(background_image_path)

    if processed_image is None or overlay_image is None or background_image is None:
        return

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
    processed_face_roi_resized = cv2.resize(processed_face_roi, (overlay_right - overlay_left, overlay_bottom - overlay_top))

    # Calculate a mask that represents the alpha channel for the processed face ROI
    mask = np.zeros(processed_face_roi_resized.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, (mask.shape[1] // 2, mask.shape[0] // 2), ((mask.shape[1] - 1) // 2, (mask.shape[0] - 1) // 2), 0, 0, 360, 255, -1)

    # Perform seamless cloning to blend the processed face onto the overlay image without visible border lines
    blended_overlay_image = cv2.seamlessClone(processed_face_roi_resized, overlay_image, mask, ((overlay_left + overlay_right) // 2, (overlay_top + overlay_bottom) // 2), cv2.NORMAL_CLONE)

    # Perform image blending of the blended overlay image with the background
    final_image = cv2.addWeighted(blended_overlay_image, alpha, background_image, 1 - alpha, 0)

    # Save the final wrapped and blended image
    cv2.imwrite(save_path, final_image)

    
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

            if processed_image is None:
                return jsonify({'error': 'Face not detected'})

            # If face, eyes, nose, and mouth are detected, send the processed image and save it to disk
            _, buffer = cv2.imencode('.png', processed_image)
            encoded_image = base64.b64encode(buffer).decode('utf-8')

            # Save the processed image with detected face, eyes, nose, and mouth
            processed_image_path = '/Users/arshitarora/Antier-Sol/face-detection/test images/processed_image_with_detections.png'
            save_image_with_detection(processed_image, processed_image_path)

            # Replace this with the actual file path of your overlay image
            overlay_image_path = '/Users/arshitarora/Antier-Sol/face-detection/test images/std.jpg'
            
            # Replace this with the actual file path of your background image
            background_image_path = '/Users/arshitarora/Antier-Sol/face-detection/test images/BaseColor.jpg'

            # Wrap and blend the images and save the updated wrapped and blended image
            wrapped_and_blended_save_path = '24.png'
            wrap_and_blend_images(processed_image_path, overlay_image_path, background_image_path, wrapped_and_blended_save_path)

            # Read the wrapped and blended image and convert it to base64 for sending in the response
            wrapped_and_blended_image = cv2.imread(wrapped_and_blended_save_path)
            _, buffer = cv2.imencode('.png', wrapped_and_blended_image)
            encoded_wrapped_and_blended_image = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'message': 'Image wrapped and blended and saved', 'wrapped_and_blended_image': encoded_wrapped_and_blended_image})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
