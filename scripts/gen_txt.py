import os
import random
import shutil
from tqdm import tqdm


def count_files(directory):
    # return len(os.listdir(directory))
    return 8  # HARD CODE for saving time


def collect_directory_info(root_directory, train_file, val_file, val_portion):
    train = open(train_file, 'a')
    val = open(val_file, 'a')
    for subdir in tqdm(os.listdir(root_directory), desc='gen txt'):
        subdir_path = os.path.join(root_directory, subdir)
        if os.path.isdir(subdir_path):
            if '_flipped' not in subdir and '_v05' not in subdir and '_v15' not in subdir:  # Flipped and original video need to be put in the same mode
                # print(root)
                file_count = count_files(subdir_path)
                line = f"{subdir} {file_count} {subdir[-1]}\n"
                # line += f"{subdir.split('/')[-1]}_v05 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_v15 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped_v05 {file_count} {os.path.basename(subdir)[3]}\n"
                # line += f"{subdir.split('/')[-1]}_flipped_v15 {file_count} {os.path.basename(subdir)[3]}\n"

                if subdir.split('_')[0].isdigit():
                    # if int(subdir.split('_')[0]) % 2 == 1:
                    #     train.write(line)
                    #     train.flush()
                    # else:
                    #     val.write(line)
                    #     val.flush()
                    train.write(line)
                    train.flush()
                else:
                    if "3C" not in subdir and "8B" not in subdir:
                        train.write(line)
                        train.flush()
                    else:
                        val.write(line)
                        val.flush()

                # if random.random() > val_portion:
                #     train.write(line)
                #     train.flush()
                # else:
                #     val.write(line)
                #     val.flush()

    train.close()
    val.close()


# Provide the root directory and output file path
root_directory = 'datasets_tsm/combined_ogcio_yanchai_06082024/images'
train_file = 'datasets_tsm/combined_ogcio_yanchai_06082024/train.txt'
val_file = 'datasets_tsm/combined_ogcio_yanchai_06082024/val.txt'
val_portion = 0.1

if os.path.exists(train_file):  # Check if the file exists
    os.remove(train_file)  # Remove the file
if os.path.exists(val_file):  # Check if the file exists
    os.remove(val_file)  # Remove the file
collect_directory_info(root_directory, train_file, val_file, val_portion)
print("Done generate txt files")
