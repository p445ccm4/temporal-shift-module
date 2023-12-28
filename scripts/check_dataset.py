import os

import cv2
from tqdm import tqdm

# Define the root folder
root_folder = "bb-dataset-cropped-upper/images"

# Iterate over the folders in the root folder
for folder in tqdm(os.listdir(root_folder), desc='checking dataset'):
    # Define the folder path
    folder_path = os.path.join(root_folder, folder)

    list_folders = os.listdir(folder_path)

    # Check if the folder is empty
    if len(list_folders) == 0:
        print(f"Folder {folder} is empty.")
        continue

    # Get the size of the first image in the folder
    image_path = os.path.join(folder_path, list_folders[0])
    image = cv2.imread(image_path)
    image_size = image.shape

    # Iterate over the other images in the folder
    for filename in list_folders:
        # Define the image path
        image_path = os.path.join(folder_path, filename)

        # Check if the image is corrupted
        try:
            image = cv2.imread(image_path)
        except cv2.error:
            print(f"Image {filename} in folder {folder} is corrupted.")
            continue

        # Check if the image has the same size as the first image
        if image.shape != image_size:
            print(f"Image {filename} in folder {folder} has a different size.")

# Print a message to indicate that the check is complete
print("Check complete.")
