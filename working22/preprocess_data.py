# preprocess_data.py
import cv2
import os
import numpy as np

# Path to gesture data
data_dir = 'data'
preprocessed_dir = 'preprocessed_data'

# Create directory for preprocessed data
os.makedirs(preprocessed_dir, exist_ok=True)

# Define the target size for resizing images
target_size = (64, 64)

for gesture_folder in os.listdir(data_dir):
    gesture_path = os.path.join(data_dir, gesture_folder)
    preprocessed_gesture_dir = os.path.join(preprocessed_dir, gesture_folder)
    os.makedirs(preprocessed_gesture_dir, exist_ok=True)

    for img_file in os.listdir(gesture_path):
        img_path = os.path.join(gesture_path, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, target_size)
        img_normalized = img_resized / 255.0

        # Save preprocessed image
        preprocessed_img_name = os.path.join(preprocessed_gesture_dir, img_file)
        np.save(preprocessed_img_name, img_normalized)
        print(f"Preprocessed and saved {preprocessed_img_name}")
