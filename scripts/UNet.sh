accelerate launch --config_file ./config.yaml run.py \
    --split_dir ./data_split \
    --model UNet \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --noise_type bw \
    --snr -4 \
    --checkpoint_dir ./checkpoints \
    --mode train \
