export CUDA_VISIBLE_DEVICES=0 && python train.py \
    --train_config configs/train_aslfeat_base_config.yaml \
    --gl3d /data/GL3D \
    --save_dir ckpt-aslfeat-base \
    --is_training=True
