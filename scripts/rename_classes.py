import os


def rename_folders(root_folder):
    """Renames folders within a given root folder one by one.

    Args:
      root_folder: The path to the root folder containing the folders to rename.
    """
    for count, folder_name in enumerate(os.listdir(root_folder)):
        source_path = os.path.join(root_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(source_path) and folder_name.split("_")[0].isdigit():
            cls = folder_name[-1]

            if cls == '1' or cls == '2':
                new_cls = '2'
            elif cls == '3':
                new_cls = '1'
            else:
                continue

            new_folder_name = folder_name[:-1] + new_cls
            new_path = os.path.join(root_folder, new_folder_name)
            os.rename(source_path, new_path)
            print(f"Renamed: '{folder_name}' to '{new_folder_name}'")


if __name__ == "__main__":
    root_folder = input("Enter the path to the root folder: ")
    rename_folders(root_folder)
