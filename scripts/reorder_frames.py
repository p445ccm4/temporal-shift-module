import os

def rename_files(directory):
    file_list = sorted(os.listdir(directory))
    for i, file_name in enumerate(file_list, start=1):
        _, ext = os.path.splitext(file_name)
        new_name = f"img_{i:03d}{ext}"
        old_path = os.path.join(directory, file_name)
        new_path = os.path.join(directory, new_name)
        os.rename(old_path, new_path)

def rename_files_in_directories(root_directory):
    for root, _, files in os.walk(root_directory):
        if root != root_directory:  # Exclude the top-level directory
            rename_files(root)

# Provide the root directory path
root_directory = 'bb-dataset-cropped-upper/images'

rename_files_in_directories(root_directory)