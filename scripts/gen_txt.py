import os
import random

def count_files(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

def collect_directory_info(root_directory, train_file, val_file, val_portion):
    for root, _, _ in os.walk(root_directory):
        if root != root_directory:  # Exclude the top-level directory
            print(root)
            if '_flipped' not in root: # Flipped and original video need to be put in the same mode
                file_count = count_files(root)
                line = f"{root.split('/')[-1]} {file_count} {os.path.basename(root)[3]}\n"
                line += f"{root.split('/')[-1]}_flipped {file_count} {os.path.basename(root)[3]}\n"

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

collect_directory_info(root_directory, train_file, val_file, val_portion)