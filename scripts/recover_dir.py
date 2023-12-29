import os

import cv2


def recover_dir(destination_dir: str):
    root_path = 'bb-dataset-cropped-upper/images'
    v05, v15 = False, False
    if destination_dir.endswith('_v05'):
        v05 = True
        source_dir = destination_dir[:-4]
    elif destination_dir.endswith('_v15'):
        v15 = True
        source_dir = destination_dir[:-4]
    else:
        source_dir = '%s' % destination_dir

    if source_dir.endswith('_flipped'):
        source_dir = source_dir[:-8]
    else:
        source_dir = source_dir + '_flipped'

    if v05:
        source_dir = source_dir + '_v05'
    if v15:
        source_dir = source_dir + '_v15'

    for file in os.listdir(os.path.join(root_path, source_dir)):
        # Read the image and flip it
        image = cv2.imread(os.path.join(root_path, source_dir, file))
        flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip

        # cv2.imshow('org', image)
        # cv2.imshow('flipped', flipped_image)
        # cv2.waitKey(0)

        # Save the flipped image to the destination folder
        destination_file_path = os.path.join(root_path, destination_dir, file)
        cv2.imwrite(destination_file_path, flipped_image)
    print(f'Recovered {destination_dir} from {source_dir}.')


def main():
    recover_dir('cls2_vid7B-Cam003_ppl19_v05')


if __name__ == '__main__':
    main()
