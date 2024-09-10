cd ..

python3 main.py bb-dataset-cropped-upper RGB \
     --arch resnet50 --num_segments 8 \
     --gd 20 --lr 0.001 --wd 1e-4 --lr_steps 20 40 --epochs 2 \
     --batch-size 16 --dropout 0.5 --consensus_type=avg --eval-freq=1 \
     --shift --shift_div=8 --shift_place=blockres --npb