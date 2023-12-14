# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import os

import cv2
import numpy as np
import torch
import torch.utils.data as data


def norm_brightness(frame, val=125):
    # Splitting into HSV
    h, s, v = cv2.split(cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))

    # Normalizing the brightness
    v = cv2.normalize(v, None, alpha=0, beta=val, norm_type=cv2.NORM_MINMAX)

    # Conver back into HSV
    hsv = cv2.merge((h, s, v))

    # Convert into color img
    res = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Return coloured image
    return res


def resize_and_pad_images(image_list, desired_height, desired_width):
    # Get the number of images
    num_images = len(image_list)

    # Create an array to store the resized and padded images
    resized_padded_images = np.zeros((num_images, desired_height, desired_width, 3), dtype=np.uint8)

    # Resize and pad each image
    for i, image in enumerate(image_list):
        image = image.astype(np.uint8)
        # Resize the image to the maximum size
        height, width, _ = image.shape
        scale_factor_h = desired_height / height
        scale_factor_w = desired_width / width
        scale_factor = min(scale_factor_h, scale_factor_w)
        resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

        # Calculate the padding values
        pad_height = (desired_height - resized_image.shape[0]) // 2
        pad_width = (desired_width - resized_image.shape[1]) // 2

        # Paste the resized image onto the new blank image with padding
        resized_padded_images[i, pad_height:pad_height + resized_image.shape[0],
        pad_width:pad_width + resized_image.shape[1]] = resized_image

    return resized_padded_images

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=True, dense_sample=False, twice_sample=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')

        if self.modality == 'RGBDiff':
            self.new_length += 1  # Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        try:
            # return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
            return cv2.imread(os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
        except Exception:
            print('error loading image:', os.path.join(self.root_path, directory, self.image_tmpl.format(idx)))
            # return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(1))).convert('RGB')]
            return cv2.imread(os.path.join(self.root_path, directory, self.image_tmpl.format(1)))

    def _parse_list(self):
        # check the frame number is large >3:
        tmp = [x.strip().split(' ') for x in open(self.list_file)] # [[path, n_frame, label], [path, n_frame, label], ...]
        if not self.test_mode or self.remove_missing:
            tmp = [item for item in tmp if int(item[1]) >= self.num_segments]
        self.video_list = [VideoRecord(item) for item in tmp]

        print('video number:%d' % (len(self.video_list)))

    def __getitem__(self, index):
        record_idx = 0

        while self.video_list[record_idx].num_frames-self.num_segments+1 < index:
            index -= self.video_list[record_idx].num_frames-self.num_segments+1
            record_idx += 1
            
        record = self.video_list[record_idx]

        indices = range(index, index+self.num_segments)

        return self.get(record, indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                img = norm_brightness(seg_imgs, 225)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
                images.append(img)
                if p < record.num_frames:
                    p += 1

        images = resize_and_pad_images(images, 224, 224).astype(np.float32)
        images /= 255.0
        images = images.transpose([0, 3, 1, 2])
        process_data = torch.from_numpy(images)
        return process_data, record.label

    def __len__(self):
        return sum([record.num_frames - self.num_segments + 1 for record in self.video_list])