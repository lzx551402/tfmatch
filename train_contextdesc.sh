export CUDA_VISIBLE_DEVICES=4 && python train.py \
    --train_config configs/train_contextdesc_config.yaml \
    --gl3d /data/GL3D \
    --save_dir ckpt-contextdesc \
    --is_training=True --device_idx 0 --data_split comb_full