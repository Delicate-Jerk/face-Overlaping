# 1st Image Overlapping 2nd Image

🎯 This is a Flask web application that detects facial features (eyes, nose, and mouth) in an image, overlaps a second image without rectangles around the features, and saves the updated overlapped image.

🔍 **Objective:**

The main objective of this application is to overlap a second image onto the first image, considering the facial features detected in the first image. The second image serves as the background, and the first image is overlaid on top, ensuring the facial features align properly.

🛠️ **Technologies and Libraries:**

- Python
- Flask
- OpenCV (cv2)

🖼️ **How It Works:**

1. The user uploads an image through the `/process_image` endpoint.
2. The application detects facial features (eyes, nose, and mouth) in the uploaded image using OpenCV's Haar cascades.
3. The first image (processed image) is then overlapped onto a second image (background image) without any rectangles around the detected facial features.
4. The updated overlapped image is saved and returned as a response.

⚙️ **Setup and Usage:**

1. Clone this repository to your local machine.
2. Install the required libraries listed in `requirements.txt`.
3. Run the Flask application using `python app.py`.
4. Use Postman or any API testing tool to send a POST request to `http://localhost:5000/process_image` with the image file attached.

📂 **Directory Structure:**

- `app.py`: Contains the Flask web application code.
- `haarcascades/`: Directory containing the Haar cascade classifiers for face, eyes, nose, and mouth detection.
- `Unknown.jpg`: The second image (background image) you want to overlap with the first image.

📷 **Sample Images:**

The sample images used in this repository are for demonstration purposes only. Feel free to replace `Unknown.jpg` with your own second image for experimentation.

🚨 **Important Note:**

- The application assumes that the uploaded image contains at least one face. If no face is detected, it will return an error.
- For best results, ensure that the second image (background image) has a face of similar size and orientation as the first image.

📝 **License:**

This project is licensed under the [MIT License](LICENSE).

👩‍💻 **Author:**

- [Your Name](https://github.com/yourusername)

Happy overlapping! 🎉
