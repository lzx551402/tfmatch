# Post-Conference Update

1. Training with/without rotation perturbation
Initially, both GeoDesc and ContextDesc are trained with input patches perturbed by random rotation up to 90 degrees. We then realized that, in most evaluation settings, testing images are mostly upright aligned, we hence disable the rotation perturbation during training (i.e., set ``rng_angle=5`` in [this file](https://github.com/lzx551402/tfmatch/blob/master/utils/npy_utils.py#L99)). If you want to reproduce the results reported in the paper, or your target application needs to handle large rotation changes, you may consider set ``rng_angle=90``.

2. Training loss
For training GeoDesc, we have deprecated using the geometry-constrained loss as in the orignal paper, and recommend using the structured loss with scale temperature, as used by ContextDesc. This is enabled by setting ``loss_type: 'LOG'`` in the [configuration file](https://github.com/lzx551402/tfmatch/blob/master/configs/train_geodesc_config.yaml#L26).

3. Improve ASLFeat
We have found that [circle loss](https://arxiv.org/abs/2002.10857) converges notably faster for training ASLFeat. To enable it, change ``loss_type`` to ``'CIRCLE'`` in the respective configuration files. The hyper-parameters of circle loss can be tuned [here](https://github.com/lzx551402/tfmatch/blob/master/losses.py#L176).

4. Future improvements
One promising improvement can be made by augmenting the training data by modern photo-realistic style transfer techniques, e.g., by [WCT2](https://github.com/clovaai/WCT2). This has been shown very effectvie in [R2D2](https://github.com/naver/r2d2) in order to deal with illumination changes, such as day-night image matching. We expect this technique to be particularly helpful, since the training data used by this project, i.e., GL3D, exhibits very limited lighting changes (most images are captured with well-conditioned and consistent lightning).