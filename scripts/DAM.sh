accelerate launch --config_file ./config.yaml run.py \
    --split_dir ./data_split \
    --model DACNN \
    --batch_size 64 \
    --epochs 100 \
    --lr 1e-3 \
    --noise_type emb \
    --snr_db 4 \
    --seed 3407 \
    --checkpoint_dir ./checkpoints \
    --mode train \
    
