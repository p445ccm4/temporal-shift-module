cd /media/nvidia/6561-3431/temporal-shift-module

python3 scripts/flip_dataset.py
python3 scripts/gen_txt.py
python3 scripts/reorder_frames.py