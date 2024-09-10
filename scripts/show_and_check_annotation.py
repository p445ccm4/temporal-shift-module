import shutil
import cv2
import os
from tqdm import tqdm


# Current working directory
cwd = 'datasets_tsm/yanchai_12072024/images'
# Label (0=Normal, 1=Slap Head, 2=Slap Body, 3=Shake)
label = 3


def rename_dir(old_dir, new_dir):
    while os.path.exists(new_dir):
        vid_frame, ppl_cls = new_dir.rsplit('_ppl', maxsplit=1)
        ppl, cls = ppl_cls.rsplit('_cls', maxsplit=1)
        new_dir = vid_frame + '_ppl' + str(int(ppl)+1) + '_cls' + cls

    # suffices = ["", "_v05", "_v15"]
    suffices = [""]

    for suffix in suffices:
        os.makedirs(new_dir + suffix)

        for file in os.listdir(old_dir + suffix):
            shutil.move(os.path.join(old_dir + suffix, file), os.path.join(new_dir + suffix, file))

        os.rmdir(old_dir + suffix)

        print(f'{old_dir + suffix} -> {new_dir + suffix}')


folder_list = sorted([folder for folder in os.listdir(cwd) if not folder.endswith('_v05') and not folder.endswith('_v15') and folder[-1] == str(label)])
# Loop through each folder in the current working directory
for folder in tqdm(folder_list, desc="Check annotations"):
    folder_path = os.path.join(cwd, folder)
    # Check if the folder is a directory
    if os.path.isdir(folder_path):
        # print(folder)

        while True:
            # Loop through each image file in the folder
            for image_file in sorted(os.listdir(folder_path)):
                # Check if the file is a JPG image
                if image_file.endswith(".jpg"):
                    # Get the path to the image file
                    image_path = os.path.join(folder_path, image_file)

                    # Read the image using OpenCV
                    image = cv2.imread(image_path)

                    # Display the image using OpenCV
                    cv2.imshow(f'{folder}: Class {folder[-1]}', image)
                    cv2.waitKey(1000 // 20)

            key = cv2.waitKey(0)
            if key == ord(' '):  # Correct / Neutral
                break
            elif key == ord('r'):  # replay
                continue
            elif key == ord('0'):
                new_name = os.path.join(cwd, folder[:-1] + '0')
                rename_dir(folder_path, new_name)
                break
            elif key == ord('1'):
                new_name = os.path.join(cwd, folder[:-1] + '1')
                rename_dir(folder_path, new_name)
                break
            elif key == ord('2'):
                new_name = os.path.join(cwd, folder[:-1] + '2')
                rename_dir(folder_path, new_name)
                break
            elif key == ord('3'):
                new_name = os.path.join(cwd, folder[:-1] + '3')
                rename_dir(folder_path, new_name)
                break
            elif key == 255:  # delete
                print("removed " + folder_path)
                shutil.rmtree(folder_path)
                break
            elif key == ord('q'):
                exit(0)

        cv2.destroyAllWindows()