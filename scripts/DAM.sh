accelerate launch --config_file ./config.yaml run.py \
    --split_dir ./data_split \
    --model Seq2Seq2 \
    --batch_size 32 \
    --epochs 100 \
    --lr 1e-3 \
    --noise_type emb \
    --snr_db -4 \
    --checkpoint_dir ./checkpoints \
    --mode train \
