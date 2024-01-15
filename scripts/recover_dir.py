import os

import cv2

from scripts.aug_hsv_dataset import hsv_augmentation


def recover_dir(destination_dir: str):
    root_path = 'bb-dataset-cropped-upper/images'
    v = 1
    if destination_dir.endswith('_v05'):
        v = 0.5
        source_dir = destination_dir[:-4]
    elif destination_dir.endswith('_v15'):
        v = 1.5
        source_dir = destination_dir[:-4]
    else:
        v = 1 / 0.5
        source_dir = destination_dir + '_v05'

    os.makedirs(os.path.join(root_path, destination_dir), exist_ok=True)

    for file in os.listdir(os.path.join(root_path, source_dir)):
        # Read the image and flip it
        image = cv2.imread(os.path.join(root_path, source_dir, file))
        # flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip
        flipped_image = hsv_augmentation(image, v)

        # cv2.imshow('org', image)
        # cv2.imshow('flipped', flipped_image)
        # cv2.waitKey(0)

        # Save the flipped image to the destination folder
        destination_file_path = os.path.join(root_path, destination_dir, file)
        cv2.imwrite(destination_file_path, flipped_image)
    print(f'Recovered {destination_dir} from {source_dir}.')


def main():
    recover_dir('cls0_vid5C-Cam003_ppl11')


if __name__ == '__main__':
    main()
