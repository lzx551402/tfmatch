export CUDA_VISIBLE_DEVICES=0 && python train.py \
    --train_config configs/train_geodesc_config.yaml \
    --gl3d /data/GL3D \
    --save_dir ckpt-geodesc \
    --is_training=True