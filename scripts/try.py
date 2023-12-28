import os

import cv2

root_path = 'bb-dataset-cropped-upper/images'
source_dir = 'cls1_vid1001_ppl0_flipped_v05'
destination_dir = 'cls1_vid1001_ppl0_v05'

for file in os.listdir(os.path.join(root_path, source_dir)):
    # Read the image and flip it
    image = cv2.imread(os.path.join(root_path, source_dir, file))
    flipped_image = cv2.flip(image, 1)  # 1 for horizontal flip

    cv2.imshow('org', image)
    cv2.imshow('flipped', flipped_image)
    cv2.waitKey(0)

    # Save the flipped image to the destination folder
    destination_file_path = os.path.join(root_path, destination_dir, file)
    cv2.imwrite(destination_file_path, flipped_image)
    # print(destination_file_path)
