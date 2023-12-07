cd /media/nvidia/6561-3431/temporal-shift-module

python3 test_models.py bb-dataset-cropped \
    --weights=checkpoint/TSM_bb-dataset-cropped_RGB_resnet50_shift8_blockres_avg_segment8_e5/ckpt.pth.tar \
    --test_segments=8 --test_crops=1 \
    --batch_size=32 \
    --csv_file=checkpoint/test_result.csv \
    --test_list=bb-dataset-cropped/train.txt