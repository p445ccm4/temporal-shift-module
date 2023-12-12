import os
import cv2
import shutil

def flip_images(source_dir, destination_dir):
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Get the path of the current file
            source_path = os.path.join(root, file)

            # Create the corresponding destination directory structure
            relative_path = os.path.relpath(source_path, source_dir)
            destination_dir_path = os.path.dirname(relative_path)
            destination_dir_name = os.path.basename(destination_dir_path) + "_flipped"
            destination_dir_path = os.path.join(destination_dir, destination_dir_name)
            os.makedirs(destination_dir_path, exist_ok=True)

            # Read the image and flip it
            image = cv2.imread(source_path)
            flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip

            # Save the flipped image to the destination folder
            destination_file_path = os.path.join(destination_dir_path, file)
            cv2.imwrite(destination_file_path, flipped_image)

# Example usage
source_directory = "bb-dataset-cropped-upper/images"
destination_directory = "bb-dataset-cropped-upper/images_flipped"

flip_images(source_directory, destination_directory)