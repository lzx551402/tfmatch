# TFMatch: Learning-based image matching in TensorFlow

## About

**TFMatch** provides a code base that tackles learning-based image matching problems, including local feature detectors and local feature descriptors. Research works below are implemented in this project:

|Method          |Reference                                           |Code |
|:--------------:|:--------------------------------------------------:|:---:|
|GeoDesc         |[Link](https://arxiv.org/abs/1807.06294), ECCV'18|[Link](https://github.com/lzx551402/geodesc)|
|ContextDesc     |[Link](https://arxiv.org/abs/1904.04084), CVPR'19|[Link](https://github.com/lzx551402/contextdesc)|
|ASLFeat         |[Link](https://arxiv.org/abs/2003.10071), CVPR'20|[Link](https://github.com/lzx551402/ASLFeat)|

## Usage

All above works use the same training data from [GL3D](https://github.com/lzx551402/GL3D). It is recommended to use this project for model training, then use the scripts provided in the respective project for model evaluation.

### Data preparation 

To prepare the training data, please refer to the instructions in [GL3D](https://github.com/lzx551402/GL3D) to download necessary geometric labels, including:

```bash
# undistorted images (62G)
bash download_data.sh gl3d_imgs 0 125
# camera parameters (<0.1G)
bash download_data.sh gl3d_cams 0 0
# SIFT keypoints (28G)
bash download_data.sh gl3d_kpts 0 57
# SIFT correspondences (6.1G)
bash download_data.sh gl3d_corr 0 12
# depth maps from MVS (30G)
bash download_data.sh gl3d_depths 0 59
```

To enrich the data diversity, you may also download [phototourism data](https://github.com/lzx551402/GL3D/blob/v2/docs/tourism_data.md). You may alternatively use the blended images and rendered depths for training ASLFeat, which provides better geometric alignment, to do so:

```bash
# blended images (58G)
bash download_data.sh gl3d_blended_images 0 117
# rendered depth maps (30G)
bash download_data.sh gl3d_rendered_depths 0 59
```

### Data parsing

After unzipping the downloaded data, you are ready to train GeoDesc and ASLFeat. Please be noded that, for the first time you run into any training, the program will need to parse the dataset and generate a set of intermediate data samples, which we refer to as ``match sets``. 

Concretely, it will create a folder named ``match_set`` in each project folder, e.g., ``GL3D/data/000000000000000000000000/match_sets``, with a set of ``*.match_set`` files that specifies the indexes of an image pair, the correspondences (for GeoDesc and ContextDesc), camera intrinsics and relative pose (for ASLFeat). Be noted that the data preparation may take around 2.5 hours.

After then, a lock file will be generated in the project folder (i.e., ``.complete``), and later training will skip this step as long as the lock file is found. To regenerate the match sets, pass ``--regenerate`` when calling the training script.

The data sampling is also conducted when generating match sets, by checking such as the number of correspondences or the rotation difference between cameras. Please refer to ``preprocess.py`` and ``tools/io.py`` for details.

### Training GeoDesc

To train GeoDesc, you may checkout ``configs/train_geodesc_config.yaml``, set up the data root, model root and query GPU index in ``train_geodesc.sh``, then call:

```bash
sh train_geodesc.sh
```

The model will be saved in ``ckpt-geodesc``.

### Training ASLFeat

As described in the paper, we conduct a two-stage training for ASLFeat. First, you may configure ``configs/train_aslfeat_base_config.yaml`` and ``train_aslfeat_base.sh``, and obtain the base model of ASLFeat without DCNs by calling:

```bash
sh train_aslfeat_base.sh
```

The model will be saved in ``ckpt-aslfeat-base``. Next, you may configure ``configs/train_aslfeat_dcn_config.yaml`` and ``train_aslfeat_dcn.sh``, and obtain the final model of ASLFeat with DCNs by calling:

```bash
sh train_aslfeat_dcn.sh
```

The model will be saved in ``ckpt-aslfeat-dcn``. To use blended images and rendered depths, pass ``--data_split blendedmvg`` when calling the training script.

### Training ContextDesc

To train ContextDesc, you will need to additionally prepare the regional features. Limited by the storage, we do not host those feature files on AWS, but provide here an [instruction](docs/extract_delf_regional_features.md) for data preparation.

Once the regional features are extracted, you may checkout ``config/train_contextdesc_config.yaml`` and ``train_contextdesc.sh``, then call:

```bash
sh train_contextdesc.sh
```

The model will be saved in ``ckpt-contextdesc``.

## Post-conference update

After serveral iterations, this project may not produce the exact results as reported in the respective papers, but overall performs better. [Here](docs/postconference_update.md) we list the major changes in the implementation and the insight behind.

## Future Support

I will have limited access to maintain this project as I am about to graduate. If you encounter any problem, feel free to open an issue or PR, and I may have other labmates to help to review.