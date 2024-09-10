import os
import shutil

# Get the current working directory
cwd = os.path.join('bb-dataset-cropped-upper/images')

# Specify the destination directory
destination_directory = os.path.join('bb-dataset-cropped-upper/images_old')

# Create the destination directory if it doesn't exist
if not os.path.exists(destination_directory):
    os.makedirs(destination_directory)

# Iterate over all the files and directories in the current working directory
for item in os.listdir(cwd):
    # Check if the item is a directory
    if os.path.isdir(os.path.join(cwd, item)):
        # Check if the directory name does not contain "Cam"
        if "Cam" not in item and "cam" not in item:
            # Move the directory to the destination directory
            shutil.move(os.path.join(cwd, item), os.path.join(destination_directory, item))

# Print a message to indicate that the operation is complete
print("Folders not containing 'Cam' have been moved to", destination_directory)