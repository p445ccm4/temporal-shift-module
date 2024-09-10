import os
import cv2
from tqdm.contrib.concurrent import process_map
# from recover_dir import recover_dir

# Define the root folder
root_folder = "datasets_tsm/combined_ogcio_yanchai_18072024/images"

org_list = os.listdir(root_folder)


# Define a function to check a single folder
def check_folder(folder):
    # Define the folder path
    folder_path = os.path.join(root_folder, folder)

    if not os.path.exists(folder_path):
        print(f"Folder {folder} is not exist.")
        # os.mkdir(folder_path)
        # recover_dir(folder)
        return

    list_folders = os.listdir(folder_path)

    # Check if the folder is empty
    if len(list_folders) == 0:
        print(f"Folder {folder} is empty.")
        # recover_dir(folder)
        # list_folders = os.listdir(folder_path)
        return

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
        except Exception as e:
            print(e)
            print(f"Image {filename} in folder {folder} is corrupted.")
            # recover_dir(folder)
            # image = cv2.imread(image_path)
            return

        if image is None:
            print(f"Image {filename} in folder {folder} is corrupted.")
            # recover_dir(folder)
            # image = cv2.imread(image_path)
            return

        try:
            # Check if the image has the same size as the first image
            if image.shape != image_size:
                print(f"Image {filename} in folder {folder} has a different size.")
                # recover_dir(folder)

                # # Get the size of the first image in the folder
                # first_image_path = os.path.join(folder_path, list_folders[0])
                # first_image = cv2.imread(first_image_path)
                # image_size = first_image.shape
                return

        except Exception as e:
            print(e)
            print(folder)


# Map the check_folder function to each folder in the root folder
process_map(check_folder, org_list, chunksize=1)

# Print a message to indicate that the check is complete
print("Check complete.")
