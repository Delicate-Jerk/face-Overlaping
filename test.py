import os

file_path = '/opt/homebrew/lib/python3.11/site-packages/cv2/data/haarcascade_mcs_nose.xml'

if os.path.exists(file_path):
    print(f"The file '{file_path}' exists.")
else:
    print(f"The file '{file_path}' does not exist.")
