cd '/media/nvidia/6561-3431/temporal-shift-module'

# make sure the dataset directory is correct before running each script
# hsv augmentation
#python3 scripts/aug_hsv_dataset.py
# rename frames starting from "img001.jpg" and make sure the numbers are continuous
# can skip this script if all data comes from step 2.
# python3 scripts/reorder_frames.py
# check for empty, corrupted folders
python3 scripts/check_dataset.py
# generate train.txt and val.txt
python3 scripts/gen_txt.py

echo 'finish dataset augmentation and checking'

#bash scripts/train_tsm_bb_upper.sh