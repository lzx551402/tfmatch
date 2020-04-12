#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Model architectures.
"""
import tensorflow as tf
import numpy as np

from losses import make_structured_loss, make_quadruple_loss, make_detector_loss
from utils.tf_utils import apply_patch_pert, apply_coord_pert, photometric_augmentation
from utils.tf_geom import get_warp, get_dist_mat, interpolate
from utils.npy_utils import get_rnd_homography, get_rnd_affine

from tools.common import Notify
from tools.io import load_pfm
from cnn_wrapper import helper
import cnn_wrapper.spatial_transformer as st

from cnn_wrapper.descnet import GeoDesc
from cnn_wrapper.aslfeat import ASLFeat
from cnn_wrapper.augdesc import VisualContext, MatchabilityPrediction, LightContextNormalization

FLAGS = tf.compat.v1.app.flags.FLAGS

DB_IMG_SIZE = 1000


def preprocess(img, kpt_coeff, spec, num_corr, photaug, pert_homo, pert_affine, dense_desc, name):
    """
    Data Preprocess.
    """
    with tf.name_scope(name):  # pylint: disable=not-context-manager
        img = tf.cast(img, tf.float32)
        img.set_shape((spec.batch_size,
                       img.get_shape()[1].value,
                       img.get_shape()[2].value,
                       img.get_shape()[3].value))
        if FLAGS.is_training and photaug:
            print(Notify.WARNING, 'Applying photometric augmentation.', Notify.ENDC)
            img = tf.map_fn(photometric_augmentation, img, back_prop=False)
        img = tf.clip_by_value(img, 0, 255)
        # perturb patches and coordinates.
        pert_kpt_affine, kpt_ncoords = apply_patch_pert(
            kpt_coeff, pert_affine, spec.batch_size, num_corr, adjust_ratio=1. if dense_desc else 5. / 6.)
        if dense_desc:
            # image standardization
            mean, variance = tf.nn.moments(
                tf.cast(img, tf.float32), axes=[1, 2], keep_dims=True)
            out = tf.nn.batch_normalization(
                img, mean, variance, None, None, 1e-5)
        else:
            # patch sampler.
            patch = st.transformer_crop(
                img, pert_kpt_affine, spec.input_size, True)
            # patch standardization
            mean, variance = tf.nn.moments(patch, axes=[1, 2], keep_dims=True)
            out = tf.nn.batch_normalization(
                patch, mean, variance, None, None, 1e-5)
        out = tf.stop_gradient(out)
        return out, kpt_ncoords, pert_homo


def feat_tower(net_input, kpt_ncoords,
               num_corr, reuse, is_training, batch_size, config, idx=0):
    with tf.name_scope('feat_tower%s' % idx):  # pylint: disable=not-context-manager
        if config['dense_desc']:
            config_dict = {}
            config_dict['interpolate'] = interpolate
            config_dict['deform_desc'] = config['deform_desc']
            config_dict['det_config'] = config['det']
            feat_tower = ASLFeat({'data': net_input, 'kpt_coord': kpt_ncoords},
                                 is_training=is_training, reuse=reuse, **config_dict)
        else:
            feat_tower = GeoDesc({'data': net_input},
                                 is_training=is_training, reuse=reuse)
    # output_name = 'conv1_up'
    output_name = 'conv6'
    feat = feat_tower.get_output_by_name(output_name)
    desc_dim = feat.get_shape()[-1]
    if config['det']['weight'] > 0:
        if is_training:
            feat = tf.reshape(
                feat, [batch_size, config['det']['kpt_n'], desc_dim])
        else:
            feat = tf.reshape(feat, [batch_size, -1, desc_dim])
    else:
        feat = tf.reshape(feat, [batch_size, num_corr, desc_dim])

    return feat_tower, feat


def aug_tower(feat_tower, kpt_ncoords, feat, img_feat,
              num_corr, reuse, is_training, batch_size, config,
              pert_homo=None, idx=0):
    aug_feat = [feat]

    with tf.name_scope('aug_tower%s' % idx):  # pylint: disable=not-context-manager
        if config['aug']['vis_context']:
            points = kpt_ncoords

            with tf.compat.v1.variable_scope('vis_context'):
                vis_context_tower = VisualContext(
                    {'img_feat': img_feat, 'local_feat': feat, 'kpt_ncoords': points},
                    is_training=is_training, reuse=reuse)
                vis_feat = vis_context_tower.get_output()
                aug_feat.append(vis_feat)

        kpt_m = None

        if config['aug']['geo_context']:
            with tf.compat.v1.variable_scope('kpt_m'):
                inter_feat = feat_tower.get_output_by_name('conv5')
                if config['dense_desc']:
                    inter_feat = tf.reshape(
                        inter_feat, [batch_size * num_corr, 1, 1, inter_feat.get_shape()[-1].value])
                kpt_m_tower = MatchabilityPrediction({'data': inter_feat},
                                                     is_training=is_training, reuse=reuse)
                kpt_m = tf.reshape(kpt_m_tower.get_output_by_name(
                    'kpt_m'), [batch_size, num_corr, 1])

            with tf.compat.v1.variable_scope('geo_context'):
                if pert_homo is not None:
                    pert_kpt_coord = apply_coord_pert(
                        kpt_ncoords, pert_homo, batch_size, num_corr)
                else:
                    pert_kpt_coord = kpt_ncoords

                kpt_m_rescale = tf.reshape(kpt_m_tower.get_output(), [
                                           batch_size, num_corr, 1])
                points = tf.concat(
                    [kpt_m_rescale, pert_kpt_coord], axis=2)

                geo_context_tower = LightContextNormalization({'points': tf.expand_dims(points, axis=2)},
                                                              is_training=is_training, reuse=reuse)
                geo_feat = geo_context_tower.get_output()
                aug_feat.append(geo_feat)

    aug_feat = tf.add_n(aug_feat)
    aug_feat = tf.nn.l2_normalize(aug_feat, axis=-1, name='l2norm')
    return aug_feat, kpt_m


def training(match_set_list, img_list, depth_list, reg_feat_list, config):
    """Build training architecture.
    Args:
        samples: List of samples.
        img_list: List of image paths.
        depth_list: List of depth paths.
        reg_feat_list: List of regional features.
        config: Configuration file.
    Returns:
        endpoints: Retured tensor list.
    """
    spec = helper.get_data_spec(model_class=GeoDesc)

    with tf.device("CPU:0"):
        batch_tensors = _training_data_queue(
            spec, match_set_list, img_list, depth_list, reg_feat_list, config)
    if config['aug']['vis_context']:
        img0, img1, depth0, depth1, kpt_coeff0, kpt_coeff1, \
            inlier_num, ori_img_size0, ori_img_size1, K0, K1, _, rel_pose, \
            img_feat0, img_feat1 = batch_tensors
    else:
        img0, img1, depth0, depth1, kpt_coeff0, kpt_coeff1, \
            inlier_num, ori_img_size0, ori_img_size1, K0, K1, _, rel_pose = batch_tensors
        img_feat0 = img_feat1 = None

    if config['use_corr_n'] > 0:
        assert config['use_corr_n'] < FLAGS.num_corr
        num_corr = config['use_corr_n']
        kpt_coeff0 = kpt_coeff0[:, 0:num_corr * 6]
        kpt_coeff1 = kpt_coeff1[:, 0:num_corr * 6]
        inlier_num = tf.minimum(inlier_num, num_corr)
        print(Notify.WARNING, '# Correspondence used in training', num_corr, Notify.ENDC)
    else:
        num_corr = FLAGS.num_corr

    inlier_mask = []
    for i in range(spec.batch_size):
        inlier_mask.append(
            tf.concat([tf.ones(inlier_num[i]), tf.zeros(num_corr - inlier_num[i])], axis=0))
    inlier_mask = tf.cast(tf.stack(inlier_mask, axis=0), tf.bool)

    # generate random affine/homography transformations
    pert_homo = tf.numpy_function(
        get_rnd_homography, [2, spec.batch_size, 0.15], tf.float32)
    pert_homo = tf.reshape(pert_homo, (2, spec.batch_size, 3, 3))

    pert_affine = tf.numpy_function(
        get_rnd_affine, [2, spec.batch_size, num_corr], tf.float32)
    pert_affine = tf.reshape(pert_affine, (2, spec.batch_size, num_corr, 3, 3))

    net_input0, kpt_ncoords0, pert_homo0 = preprocess(
        img0, kpt_coeff0, spec, num_corr, config['photaug'],
        pert_homo[0], pert_affine[0], config['dense_desc'], name='input0')
    net_input1, kpt_ncoords1, pert_homo1 = preprocess(
        img1, kpt_coeff1, spec, num_corr, config['photaug'],
        pert_homo[1], pert_affine[1], config['dense_desc'], name='input1')

    feat_tower0, feat0 = feat_tower(
        net_input0, kpt_ncoords0, num_corr,
        False, FLAGS.is_training, spec.batch_size, config, idx=0)

    aug_feat0, kpt_m0 = aug_tower(
        feat_tower0, kpt_ncoords0, feat0, img_feat0, num_corr,
        False, FLAGS.is_training, spec.batch_size, config,
        pert_homo=pert_homo0, idx=0)

    feat_tower1, feat1 = feat_tower(
        net_input1, kpt_ncoords1, num_corr,
        True, FLAGS.is_training, spec.batch_size, config, idx=1)

    aug_feat1, kpt_m1 = aug_tower(
        feat_tower1, kpt_ncoords1, feat1, img_feat1, num_corr,
        True, FLAGS.is_training, spec.batch_size, config,
        pert_homo=pert_homo1, idx=1)

    endpoints = {}
    with tf.name_scope('loss'):  # pylint: disable=not-context-manager
        loss_type = config['loss_type']

        structured_loss = tf.constant(0.)
        accuracy = tf.constant(0.)

        if config['det']['weight'] > 0:
            def _grid_positions(h, w, bs):
                w = tf.cast(w, tf.int32)
                h = tf.cast(h, tf.int32)
                x_rng = tf.range(0, w)
                y_rng = tf.range(0, h)
                xv, yv = tf.meshgrid(x_rng, y_rng)
                return tf.cast(tf.tile(tf.reshape(tf.stack((yv, xv), axis=-1), (1, -1, 2)), (bs, 1, 1)), tf.float32)

            dense_feat_map0, score_map0 = feat_tower0.endpoints
            dense_feat_map1, score_map1 = feat_tower1.endpoints

            cur_feat_size0 = tf.constant(
                [score_map0.get_shape()[1].value, score_map0.get_shape()[2].value], dtype=tf.float32)
            cur_feat_size1 = tf.constant(
                [score_map1.get_shape()[1].value, score_map1.get_shape()[2].value], dtype=tf.float32)

            pos0 = _grid_positions(
                cur_feat_size0[0], cur_feat_size0[1], spec.batch_size)

            r0 = ori_img_size0 / cur_feat_size0[::-1]
            r1 = ori_img_size1 / cur_feat_size1[::-1]
            r_K0 = tf.stack([K0[:, 0] / r0[:, 0][..., None], K0[:, 1] /
                             r0[:, 1][..., None], K0[:, 2]], axis=1)
            r_K1 = tf.stack([K1[:, 0] / r1[:, 0][..., None], K1[:, 1] /
                             r1[:, 1][..., None], K1[:, 2]], axis=1)

            pos0, pos1, _ = get_warp(
                pos0, rel_pose, depth0, r_K0, depth1, r_K1, spec.batch_size)

            det_structured_loss, det_accuracy = make_detector_loss(
                pos0, pos1, dense_feat_map0, dense_feat_map1,
                score_map0, score_map1, spec.batch_size, num_corr, loss_type, config)

            structured_loss = det_structured_loss
            accuracy = det_accuracy
        else:
            structured_loss, accuracy, _ = make_structured_loss(
                aug_feat0, aug_feat1, loss_type=loss_type,
                inlier_mask=inlier_mask, name='loss')
            if config['aug']['geo_context'] or config['aug']['vis_context']:
                if config['aug']['kpt_m'] > 0 and config['det']['weight'] < 0:
                    kpt_m_loss, kpt_m_accuracy = make_quadruple_loss(
                        kpt_m0, kpt_m1, inlier_num)
                    endpoints['kpt_m_loss'] = kpt_m_loss
                    endpoints['kpt_m_accuracy'] = kpt_m_accuracy
                raw_loss, raw_accuracy, _ = make_structured_loss(
                    tf.nn.l2_normalize(feat0, -1), tf.nn.l2_normalize(feat1, -1),
                    loss_type=loss_type, inlier_mask=inlier_mask, name='raw_loss')
                endpoints['raw_loss'] = raw_loss
                endpoints['raw_accuracy'] = raw_accuracy
            else:
                endpoints['raw_loss'] = None
                endpoints['raw_accuracy'] = None

        endpoints['structured_loss'] = structured_loss
        endpoints['accuracy'] = accuracy

    # Add summaries for viewing model statistics on TensorBoard.
    with tf.name_scope('summaries'):  # pylint: disable=not-context-manager
        scalars = [accuracy, structured_loss]
        _activation_summaries([], scalars)

    return endpoints


def _training_data_queue(spec, match_set_list, img_list, depth_list, reg_feat_list, config):
    """Queue to read training data in binary.
    Args:
        spec: Model specifications.
        match_set_list: List of samples.
        img_list: List of image paths.
        depth_list: List of depth paths.
        reg_feat_list: List of reginal features.
    Returns:
        batch_tensors: List of fetched data.
    """

    with tf.name_scope('data_queue'):  # pylint: disable=not-context-manager
        # sample queue. the sample list has been shuffled.
        def _match_set_parser(val):
            def _parse_img(img_paths, idx):
                img_path = tf.squeeze(tf.gather(img_paths, idx))
                img = tf.image.decode_image(
                    tf.io.read_file(img_path), channels=1)
                img.set_shape((DB_IMG_SIZE, DB_IMG_SIZE, 1))
                if config['resize'] > 0:
                    img = tf.image.resize(
                        img, (config['resize'], config['resize']))
                    pad_size = int(config['resize'] * 0.1)
                else:
                    pad_size = int(DB_IMG_SIZE * 0.1)
                if not config['dense_desc']:
                    # avoid boundary effect.
                    img = tf.pad(img, [[pad_size, pad_size], [pad_size, pad_size], [0, 0]],
                                 mode='SYMMETRIC')
                return img

            def _parse_depth(depth_paths, idx):
                depth = tf.numpy_function(
                    load_pfm, [tf.squeeze(tf.gather(depth_paths, idx))], tf.float32)
                depth.set_shape((DB_IMG_SIZE // 4, DB_IMG_SIZE // 4))
                target_size = DB_IMG_SIZE // 4
                if config['resize'] > 0:
                    target_size = config['resize']
                if target_size != DB_IMG_SIZE // 4:
                    depth = tf.image.resize(
                        depth[..., None], (target_size, target_size))
                    depth = tf.squeeze(depth, axis=-1)
                return depth

            def _parse_reg_feat(reg_feat_paths, idx, reg_feat_reso, reg_feat_dim):
                reg_feat_path = tf.squeeze(tf.gather(reg_feat_paths, idx))
                reg_feat = tf.decode_raw(
                    tf.read_file(reg_feat_path), tf.float32)
                reg_feat = tf.reshape(
                    reg_feat, (reg_feat_reso, reg_feat_reso, reg_feat_dim))
                return reg_feat

            decoded = tf.decode_raw(val, tf.float32)
            idx0 = tf.cast(decoded[0], tf.int32)
            idx1 = tf.cast(decoded[1], tf.int32)
            inlier_num = tf.cast(decoded[2], tf.int32)
            ori_img_size0 = tf.reshape(decoded[3:5], (2,))
            ori_img_size1 = tf.reshape(decoded[5:7], (2,))
            K0 = tf.reshape(decoded[7:16], (3, 3))
            K1 = tf.reshape(decoded[16:25], (3, 3))
            e_mat = tf.reshape(decoded[25:34], (3, 3))
            rel_pose = tf.reshape(decoded[34:46], (3, 4))
            kpt_coeff0 = tf.slice(decoded, [46], [6 * FLAGS.num_corr])
            kpt_coeff1 = tf.slice(
                decoded, [46 + 6 * FLAGS.num_corr], [6 * FLAGS.num_corr])
            # parse images.
            img0 = _parse_img(img_list, idx0)
            img1 = _parse_img(img_list, idx1)
            # parse depths
            depth0 = _parse_depth(depth_list, idx0)
            depth1 = _parse_depth(depth_list, idx1)
            fetch_tensors = [img0, img1, depth0, depth1, kpt_coeff0, kpt_coeff1, inlier_num,
                             ori_img_size0, ori_img_size1, K0, K1, e_mat, rel_pose]
            if config['aug']['vis_context']:
                reg_feat_reso = config['aug']['vis_feat_reso']
                reg_feat_dim = config['aug']['vis_feat_dim']
                # parse reginal feat.
                reg_feat0 = _parse_reg_feat(
                    reg_feat_list, idx0, reg_feat_reso, reg_feat_dim)
                reg_feat1 = _parse_reg_feat(
                    reg_feat_list, idx1, reg_feat_reso, reg_feat_dim)
                fetch_tensors.extend([reg_feat0, reg_feat1])
            return fetch_tensors

        # decoded:
        # [1] inlier_num: 1 float
        # [2] idx: 2 float
        # [3] ori_img_size0: 2 float
        # [4] ori_img_size1: 2 float
        # [5] K0: 9 float
        # [6] K1: 9 float
        # [7] e_mat: 9 float
        # [8] rel_pose: 12 float
        # [9] kpt_coeff: 1024 * 6 * 2 float
        # [10] geo_sim: 1024 float
        dataset = tf.data.FixedLengthRecordDataset(match_set_list, 53432)
        if FLAGS.is_training:
            dataset = dataset.shuffle(buffer_size=spec.batch_size * 32)
        dataset = dataset.repeat(2)
        dataset = dataset.map(
            _match_set_parser, num_parallel_calls=spec.batch_size * 2)
        dataset = dataset.batch(spec.batch_size)
        dataset = dataset.prefetch(buffer_size=spec.batch_size * 4)
        iterator = dataset.make_one_shot_iterator()
        batch_tensors = iterator.get_next()
    return batch_tensors


def _activation_summaries(histo, scalar):
    for act in histo:
        tf.summary.histogram(act.op.name + '/histogram', act)
    for act in scalar:
        tf.summary.scalar(act.op.name + '/scalar', act)
