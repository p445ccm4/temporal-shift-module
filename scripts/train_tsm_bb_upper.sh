# You should get TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth
python3 main.py bb-dataset-cropped-upper RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.005 --wd 1e-4 --lr_steps 20 40 --epochs 10 \
     --batch-size 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb \
     --tune_from=pretrained/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e50.pth