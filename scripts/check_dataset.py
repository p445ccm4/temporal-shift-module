import os
import cv2
from tqdm.contrib.concurrent import process_map
from recover_dir import recover_dir

# Define the root folder
root_folder = "bb-dataset-cropped-upper/images"

org_list = [f for f in os.listdir(root_folder) if
            not f.endswith('_v05') and not f.endswith('_v15') and not f.endswith('_flipped')]
suffices = ['', '_v05', '_v15']


# Define a function to check a single folder
def check_folder(folder):
    # Define the folder path
    folder_path = os.path.join(root_folder, folder)

    for suffix in suffices:
        abs_folder_path_with_suffix = folder_path + suffix
        rel_folder_path_with_suffix = folder + suffix

        if not os.path.exists(abs_folder_path_with_suffix):
            print(f"Folder {rel_folder_path_with_suffix} is not exist.")
            os.mkdir(abs_folder_path_with_suffix)
            recover_dir(rel_folder_path_with_suffix)

        list_folders = os.listdir(abs_folder_path_with_suffix)

        # Check if the folder is empty
        if len(list_folders) == 0:
            print(f"Folder {rel_folder_path_with_suffix} is empty.")
            recover_dir(rel_folder_path_with_suffix)
            list_folders = os.listdir(abs_folder_path_with_suffix)

        # Get the size of the first image in the folder
        image_path = os.path.join(abs_folder_path_with_suffix, list_folders[0])
        image = cv2.imread(image_path)
        image_size = image.shape

        # Iterate over the other images in the folder
        for filename in list_folders:
            # Define the image path
            image_path = os.path.join(abs_folder_path_with_suffix, filename)

            # Check if the image is corrupted
            try:
                image = cv2.imread(image_path)
            except Exception as e:
                print(e)
                print(f"Image {filename} in folder {rel_folder_path_with_suffix} is corrupted.")
                recover_dir(rel_folder_path_with_suffix)
                image = cv2.imread(image_path)

            if image is None:
                print(f"Image {filename} in folder {rel_folder_path_with_suffix} is corrupted.")
                recover_dir(rel_folder_path_with_suffix)
                image = cv2.imread(image_path)

            try:
                # Check if the image has the same size as the first image
                if image.shape != image_size:
                    print(f"Image {filename} in folder {rel_folder_path_with_suffix} has a different size.")
                    recover_dir(rel_folder_path_with_suffix)

                    # Get the size of the first image in the folder
                    first_image_path = os.path.join(abs_folder_path_with_suffix, list_folders[0])
                    first_image = cv2.imread(first_image_path)
                    image_size = first_image.shape

            except Exception as e:
                print(e)
                print(rel_folder_path_with_suffix)


# Map the check_folder function to each folder in the root folder
process_map(check_folder, org_list)

# Print a message to indicate that the check is complete
print("Check complete.")
