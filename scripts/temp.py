import os
import shutil
from tqdm import tqdm

def delete_folders(base_path):
    for foldername, subfolders, filenames in tqdm(os.walk(base_path)):
        # Avoid deleting the root folder
        if foldername == base_path:
            continue
        if len(filenames) != 8:
            # shutil.rmtree(foldername)
            print(f"Deleted: {foldername}")

# Replace 'your_directory_path' with the path to the directory you want to check
delete_folders('datasets_tsm/combined_ogcio_yanchai_06082024/images')
