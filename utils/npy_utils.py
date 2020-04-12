#!/usr/bin/env python
"""
Copyright 2018, Zixin Luo, HKUST.
Numpy utils.
"""

import os
from math import cos, sin, pi
import random

import numpy as np
import tensorflow as tf

from tools.common import Notify

FLAGS = tf.compat.v1.app.flags.FLAGS


class EndPoints(object):
    def __init__(self, endpoints, config):
        self.endpoints = endpoints
        self.config = config
        # define loss terms.
        self.loss_terms = {}
        self.loss_terms['structured_loss'] = endpoints['structured_loss']
        self.loss_terms['kpt_m_loss'] = endpoints['kpt_m_loss'] if config['aug']['geo_context'] else None
        self.loss_terms['raw_loss'] = endpoints['raw_loss'] if config['aug']['geo_context'] or config['aug']['vis_context'] else None

        # define display terms.
        self.disp_terms = {}
        self.disp_terms['Structured loss'] = endpoints['structured_loss']
        self.disp_terms['Accuracy'] = endpoints['accuracy']
        self.disp_terms['Raw loss'] = endpoints['raw_loss'] if config['aug']['geo_context'] or config['aug']['vis_context'] else None
        self.disp_terms['Raw accuracy'] = endpoints['raw_accuracy'] if config['aug']['geo_context'] or config['aug']['vis_context'] else None
        self.disp_terms['Quad loss'] = endpoints['kpt_m_loss'] if config['aug']['geo_context'] else None
        self.disp_terms['Quad accuracy'] = endpoints['kpt_m_accuracy'] if config['aug']['geo_context'] else None

        self.disp_buffer = []

    def get_total_loss(self):
        disp_str = 'Loss components:'
        tot_loss = 0
        all_keys = list(self.loss_terms.keys())
        for some_key in all_keys:
            if self.loss_terms[some_key] is not None:
                disp_str = ' '.join([disp_str, some_key])
                tot_loss += self.loss_terms[some_key]
        print(Notify.INFO, disp_str, Notify.ENDC)
        return tot_loss

    def get_print_list(self):
        disp_list = []
        all_keys = list(self.disp_terms.keys())
        for some_key in all_keys:
            if self.disp_terms[some_key] is not None:
                disp_list.append(self.disp_terms[some_key])
                self.disp_buffer.append((some_key, []))
        return disp_list

    def add_print_items(self, out_disp_terms):
        for idx, val in enumerate(out_disp_terms):
            self.disp_buffer[idx][1].append(val)

    def disp_and_clear(self):
        for idx, val in enumerate(self.disp_buffer):
            avg_val = sum(val[1]) / float(len(val[1]))
            print(Notify.INFO, (val[0] + ' = %.6f') % avg_val, Notify.ENDC)
            self.disp_buffer[idx] = (val[0], [])


def load_config(config_path):
    import json
    if os.path.exists(config_path):
        with open(config_path) as fin:
            config_dict = json.load(fin)
            print(Notify.INFO, 'Using config file', config_path, Notify.ENDC)
    else:
        print(Notify.WARNING, 'Config file does not exist', config_path, Notify.ENDC)
    return config_dict


def get_rnd_homography(tower_num, batch_size, pert_ratio=0.25):
    import cv2
    all_homo = []
    corners = np.array([[-1, 1], [1, 1], [-1, -1], [1, -1]], dtype=np.float32)
    for _ in range(tower_num):
        one_tower_homo = []
        for _ in range(batch_size):
            rnd_pert = np.random.uniform(-2 * pert_ratio, 2 * pert_ratio, (4, 2)).astype(np.float32)
            pert_corners = corners + rnd_pert
            M = cv2.getPerspectiveTransform(corners, pert_corners)
            one_tower_homo.append(M)
        one_tower_homo = np.stack(one_tower_homo, axis=0)
        all_homo.append(one_tower_homo)
    all_homo = np.stack(all_homo, axis=0)
    return all_homo.astype(np.float32)


def get_rnd_affine(tower_num, batch_size, num_corr, sync=True, distribution='uniform',
                   crop_scale=0.5, rng_angle=5, rng_scale=0.3, rng_anis=0.4):
    """In order to enhance rotation invariance, applying
    random affine transformation (3x3) on matching patches.
    Args:
        batch_size: Training batch size (number of data bags).
        num_corr: Number of correspondences in a data bag.
        crop_scale: The ratio to apply central cropping.
        rng_angle: Range of random rotation angle.
        rng_scale: Range of random scale.
        rng_anis: Range of random anis.
    Returns:
        all_pert_mat: Transformation matrices.
    """
    num_patches = batch_size * num_corr

    if sync:
        sync_angle = np.random.uniform(-90, 90, (num_patches, ))
    else:
        sync_angle = 0

    all_pert_affine = []
    # two feature towers
    for _ in range(tower_num):
        if distribution == 'uniform':
            rnd_scale = np.random.uniform(2**-rng_scale, 2**rng_scale, (num_patches, ))
            rnd_anis = np.random.uniform(np.sqrt(2**-rng_anis), np.sqrt(2**rng_anis), (num_patches, ))
            rnd_angle = np.random.uniform(-rng_angle, rng_angle, (num_patches, ))
        elif distribution == 'normal':
            rnd_scale = 1 + np.random.normal(0, rng_scale / 2 / 3, (num_patches, ))
            rnd_anis = 1 + np.random.normal(0, rng_anis / 2 / 3, (num_patches, ))
            rnd_angle = np.random.normal(0, rng_angle / 3, (num_patches, ))

        rnd_scale *= crop_scale
        rnd_angle = (rnd_angle + sync_angle) / 180. * pi

        pert_affine = np.zeros((num_patches, 9), dtype=np.float32)
        pert_affine[:, 0] = np.cos(rnd_angle) * rnd_scale * rnd_anis
        pert_affine[:, 1] = np.sin(rnd_angle) * rnd_scale * rnd_anis
        pert_affine[:, 2] = 0
        pert_affine[:, 3] = -np.sin(rnd_angle) * rnd_scale / rnd_anis
        pert_affine[:, 4] = np.cos(rnd_angle) * rnd_scale / rnd_anis
        pert_affine[:, 5] = 0
        pert_affine[:, 8] = np.ones((num_patches, ), dtype=np.float32)
        pert_affine = np.reshape(pert_affine, (batch_size, num_corr, 3, 3))
        all_pert_affine.append(pert_affine)
    all_pert_affine = np.stack(all_pert_affine, axis=0)
    return all_pert_affine.astype(np.float32)
