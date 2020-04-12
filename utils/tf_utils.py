#!/usr/bin/env python
"""
Adapted from SuperPoint:
https://github.com/rpautrat/SuperPoint
"""
import tensorflow as tf
import utils.photometric_augmentation as photaug


def photometric_augmentation(image, random_order=True):
    primitives = photaug.augmentations
    config = {}
    config['random_brightness'] = {'max_abs_change': 50}
    config['random_contrast'] = {'strength_range': [0.3, 1.5]}
    # config['additive_gaussian_noise'] = {'stddev_range': [0, 10]}
    # config['additive_speckle_noise'] = {'prob_range': [0, 0.0035]}
    # config['additive_shade'] = {'transparency_range': [-0.5, 0.5],
    # 'kernel_size_range': [100, 150]}
    config['motion_blur'] = {'max_kernel_size': 3}

    with tf.name_scope('photometric_augmentation'):
        prim_configs = [config.get(p, {}) for p in primitives]

        indices = tf.range(len(primitives))
        if random_order:
            indices = tf.random.shuffle(indices)

        def step(i, image):
            fn_pairs = [(tf.equal(indices[i], j), lambda p=p, c=c: getattr(photaug, p)(image, **c))
                        for j, (p, c) in enumerate(zip(primitives, prim_configs))]
            image = tf.case(fn_pairs)
            return i + 1, image

        _, aug_image = tf.while_loop(lambda i, image: tf.less(i, len(primitives)),
                                     step, [0, image], parallel_iterations=1)

    return aug_image


def apply_coord_pert(trans, pert_homo, batch_size, num_corr):
    tmp_ones = tf.ones((batch_size, num_corr, 1))
    homo_coord = tf.concat((trans, tmp_ones), axis=-1)
    pert_coord = tf.matmul(homo_coord, pert_homo, transpose_b=True)
    homo_scale = tf.expand_dims(pert_coord[:, :, 2], axis=-1)
    pert_coord = pert_coord[:, :, 0:2]
    pert_coord = pert_coord / homo_scale
    return pert_coord


def apply_patch_pert(kpt_param, pert_affine, batch_size, num_corr, adjust_ratio=1):
    """
    Args:
        kpt_param: 6-d keypoint parameterization
        pert_mat: 3x3 perturbation matrix.
    Returns:
        pert_theta: perturbed affine transformations.
        trans: translation vectors, i.e., keypoint coordinates.
        pert_mat: perturbation matrix.
    """
    kpt_affine = tf.reshape(kpt_param, shape=(batch_size, num_corr, 2, 3))
    rot = kpt_affine[:, :, :, 0:2]
    trans = kpt_affine[:, :, :, 2]
    # adjust the translation as input images are padded.
    trans_with_pad = tf.expand_dims(trans * adjust_ratio, axis=-1)
    kpt_affine_with_pad = tf.concat((rot, trans_with_pad), axis=-1)
    pert_kpt_affine = tf.matmul(kpt_affine_with_pad, pert_affine)
    return pert_kpt_affine, trans