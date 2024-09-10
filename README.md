# TSM with scripts for labelling and data augmentations

Originated from https://github.com/mit-han-lab/temporal-shift-module.

## Requirements

PyTorch

## Dataset Preparation

The dataset format is different from any other motion datasets, but it is close to the Kinetics dataset format.

0. If a dataset is already prepared, skip this part and go to training.
1. Prepare a video.
2. Data Labelling:

```shell
# check on script/label_tsm_data.py for class id and video path 
python3 script/label_tsm_data.py
# press 'q' to exit
# press any other keys for next frame
# dont click on the image on the first 8 frames
# afterwards, click twice to bound the target
```

3. Data Augmentations:

```shell
# make sure the dataset directory is correct before running each script
# left-right flipping
python3 scripts/flip_dataset.py
# hsv augmentation
python3 scripts/hsv_aug_dataset.py
# generate train.txt and val.txt
python3 scripts/gen_txt.py
# rename frames starting from "img001.jpg" and make sure the numbers are continuous
# can skip this script if all data comes from step 2.
python3 scripts/reorder_frames.py
```

or

```shell
bash scripts/before_train.sh
```

## Training

```shell
python3 main.py bb-dataset-cropped-upper RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 5 \
     --batch-size 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
```

or

```shell
bash scripts/train_tsm_bb_upper.sh
```

Saved checkpoints are located under "checkpoint/".

## Testing

```shell
python3 test_models.py bb-dataset-cropped-upper \
    --weights=checkpoint/TSM_bb-dataset-cropped-upper_RGB_resnet50_shift8_blockres_avg_segment8_e2_2023_12_15/ckpt.best.pth.tar \
    --test_segments=8 --test_crops=1 \
    --batch_size=32 \
    --csv_file=checkpoint/test_result.csv \
    --test_list=bb-dataset-cropped-upper/val.txt
```

or

```shell
bash scripts/test.sh
```

## Tensor RT conversion

TensorRT conversion: https://github.com/wang-xinyu/tensorrtx
YOLOv7-TSM 2-stage detection: https://github.com/p445ccm4/YOLO-TSM-2stage-detection
