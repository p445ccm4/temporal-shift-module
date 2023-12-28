import os
import random
import shutil

from tqdm import tqdm


def count_files(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count


def collect_directory_info(root_directory, train_file, val_file, val_portion):
    for root, _, _ in tqdm(os.walk(root_directory), desc='gen txt'):
        if root != root_directory:  # Exclude the top-level directory
            if '_flipped' not in root and '_v05' not in root and '_v15' not in root:  # Flipped and original video need to be put in the same mode
                # print(root)
                file_count = count_files(root)
                line = f"{root.split('/')[-1]} {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_v05 {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_v15 {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_flipped {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_flipped_v05 {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_flipped_v15 {file_count} {os.path.basename(root)[3]}\n"

                if random.random() > val_portion:
                    with open(train_file, 'a') as f:
                        f.write(line)
                else:
                    with open(val_file, 'a') as f:
                        f.write(line)


# Provide the root directory and output file path
root_directory = 'bb-dataset-cropped-upper/images'
train_file = 'bb-dataset-cropped-upper/train.txt'
val_file = 'bb-dataset-cropped-upper/val.txt'
val_portion = 0.1

source_directory = "bb-dataset-cropped-upper/images_hsv"
if os.path.exists(source_directory):
    for file in os.listdir(source_directory):
        shutil.move(os.path.join(source_directory, file), root_directory)
    os.rmdir(source_directory)

if os.path.exists(train_file):  # Check if the file exists
    os.remove(train_file)  # Remove the file
if os.path.exists(val_file):  # Check if the file exists
    os.remove(val_file)  # Remove the file
collect_directory_info(root_directory, train_file, val_file, val_portion)
print("Done generate txt files")
