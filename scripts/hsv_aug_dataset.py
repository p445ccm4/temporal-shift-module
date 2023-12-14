import os

import cv2
import numpy as np


def hsv_augmentation(image, value_scale=1):
    # Convert the BGR image to HSV format
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Apply the HSV augmentation to each channel separately
    hsv_image[..., 2] = np.clip(hsv_image[..., 2].astype(np.float32) * value_scale, 0, 255)  # Scale the value

    # Convert the HSV image back to BGR color space
    augmented_image = cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return augmented_image


def aug_images(source_dir, destination_dir, v):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Get the path of the current file
            source_path = os.path.join(root, file)

            # Create the corresponding destination directory structure
            relative_path = os.path.relpath(source_path, source_dir)
            destination_dir_path = os.path.dirname(relative_path)
            destination_dir_name = os.path.basename(destination_dir_path) + "_v" + str(v).replace('.', '')
            destination_dir_path = os.path.join(destination_dir, destination_dir_name)
            os.makedirs(destination_dir_path, exist_ok=True)

            # Read the image and flip it
            image = cv2.imread(source_path)
            flipped_image = hsv_augmentation(image, v)  # 1 for horizontal flip

            # Save the flipped image to the destination folder
            destination_file_path = os.path.join(destination_dir_path, file)
            cv2.imwrite(destination_file_path, flipped_image)


# Example usage
source_directory = "bb-dataset-cropped-upper/new_images"
destination_directory = "bb-dataset-cropped-upper/images_hsv_aug"
v_gains = [0.5, 1.5]

for v in v_gains:
    aug_images(source_directory, destination_directory, v)
