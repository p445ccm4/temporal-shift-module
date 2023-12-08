import os
import shutil
from PIL import Image

def crop_images_in_folder(folder_path, output_folder):
    for root, dirs, files in os.walk(folder_path):
        # Get the relative path from the original folder to the current subfolder
        relative_folder_path = os.path.relpath(root, folder_path)
        output_subfolder = os.path.join(output_folder, relative_folder_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for filename in files:
            if filename.endswith(".jpg") or filename.endswith(".png"):  # Adjust the file extensions as per your image types
                file_path = os.path.join(root, filename)
                img = Image.open(file_path)
                width, height = img.size

                if width < height * 0.6:  # Checking if the ratio is greater than 1:2 (0.5)
                    cropped_img = img.crop((0, 0, width, height // 2))  # Crop the top half of the image
                    output_file_path = os.path.join(output_subfolder, filename)
                    cropped_img.save(output_file_path)  # Save the cropped image in the output folder
                else:
                    output_file_path = os.path.join(output_subfolder, filename)
                    shutil.copyfile(file_path, output_file_path)  # Copy the non-cropped image to the output folder

                img.close()

# Example usage
parent_folder_path = "bb-dataset-cropped/images"
output_folder_path = "bb-dataset-cropped-upper/images"

# Create the output folder if it doesn't exist
os.makedirs(output_folder_path, exist_ok=True)

for folder in os.listdir(parent_folder_path):
    folder_path = os.path.join(parent_folder_path, folder)
    if os.path.isdir(folder_path):
        output_subfolder_path = os.path.join(output_folder_path, folder)
        crop_images_in_folder(folder_path, output_subfolder_path)