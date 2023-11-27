import os

def count_files(directory):
    file_count = 0
    for _, _, files in os.walk(directory):
        file_count += len(files)
    return file_count

def collect_directory_info(root_directory, output_file):
    with open(output_file, 'w') as f:
        for root, _, _ in os.walk(root_directory):
            file_count = count_files(root)
            line = f"{root} {file_count} {os.path.basename(root)[3]}\n"
            f.write(line)

# Provide the root directory and output file path
root_directory = 'bb-dataset-cropped/images'
output_file = 'bb-dataset-cropped/train.txt'

collect_directory_info(root_directory, output_file)