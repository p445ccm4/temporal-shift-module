import os


def rename_folders(parent_directory, video_name):
    try:
        os.chdir(parent_directory)
        folders0 = [folder for folder in os.listdir() if
                    os.path.isdir(folder) and folder.startswith('cls0')]  # cls0_vid7022_ppl0
        folders1 = [folder for folder in os.listdir() if
                    os.path.isdir(folder) and folder.startswith('cls1')]  # cls1_vid7022_ppl0
        folders2 = [folder for folder in os.listdir() if
                    os.path.isdir(folder) and folder.startswith('cls2')]  # cls2_vid7022_ppl0

        for cls, folders in enumerate([folders0, folders1, folders2]):
            for i, folder in enumerate(folders):
                new_folder_name = f'cls{cls}_vid{video_name}_ppl{i}'
                os.rename(folder, new_folder_name)
                print(f"Renamed {folder} to {new_folder_name}")

    except Exception as e:
        print(f"An error occurred: {e}")


parent_directory = "bb-dataset-cropped-upper/images_flipped/12Y-cam003"
video_name = "12Y-cam003"

rename_folders(parent_directory, video_name)
