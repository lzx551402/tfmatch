#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Training preprocessings.
"""

import os
import time
from random import shuffle, seed

import tensorflow as tf
import progressbar

from tools.common import Notify
from tools.io import parse_corr_to_match_set, read_list

FLAGS = tf.app.flags.FLAGS


def get_match_set_list(imageset_list_path, q_diff_thld, rot_diff_thld):
    """Get the path list of match sets.
    Args:
        imageset_list_path: Path to imageset list.
        q_diff_thld: Threshold of image pair sampling regarding camera orientation.
    Returns:
        match_set_list: List of match set path.
    """
    imageset_list = [os.path.join(FLAGS.gl3d, 'data', i)
                     for i in read_list(imageset_list_path)]
    print(Notify.INFO, 'Use # imageset', len(imageset_list), Notify.ENDC)
    match_set_list = []
    # discard image pairs whose image simiarity is beyond the threshold.
    for i in imageset_list:
        match_set_folder = os.path.join(i, 'match_sets')
        if os.path.exists(match_set_folder):
            match_set_files = os.listdir(match_set_folder)
            for val in match_set_files:
                name, ext = os.path.splitext(val)
                if ext == '.match_set':
                    splits = name.split('_')
                    q_diff = int(splits[2])
                    rot_diff = int(splits[3])
                    if q_diff >= q_diff_thld and rot_diff <= rot_diff_thld:
                        match_set_list.append(
                            os.path.join(match_set_folder, val))
    # ensure the testing gives deterministic results.
    if not FLAGS.is_training:
        seed(0)
    shuffle(match_set_list)
    print(Notify.INFO, 'Get # match sets', len(match_set_list), Notify.ENDC)
    return match_set_list


def prepare_match_sets(regenerate, is_training, q_diff_thld=3, rot_diff_thld=60, data_split='comb'):
    """Generate match sets from corr.bin files. Index match sets w.r.t global image index list.
    Args:
        regenerate: Flag to indicate whether to regenerate match sets.
        is_training: Use training imageset or testing imageset.
        img_sim_thld: Threshold of image pair sampling regarding image similarity.
        rot_diff_thld: Threshold of image pair sampling regarding rotation difference.
        data_list: Data split name.
    Returns:
        match_set_list: List of match sets path.
        global_img_list: List of global image path.
        global_context_feat_list:
    """
    # get necessary lists.
    gl3d_list_folder = os.path.join(FLAGS.gl3d, 'list', data_split)
    global_info = read_list(os.path.join(
        gl3d_list_folder, 'image_index_offset.txt'))
    global_img_list = [os.path.join(FLAGS.gl3d, i) for i in read_list(
        os.path.join(gl3d_list_folder, 'image_list.txt'))]
    global_reg_feat_list = [os.path.join(FLAGS.gl3d, i) for i in read_list(
        os.path.join(gl3d_list_folder, 'regional_feat_list.txt'))]
    global_depth_list = [os.path.join(FLAGS.gl3d, i) for i in read_list(
        os.path.join(gl3d_list_folder, 'depth_list.txt'))]
    lock_path = os.path.join(os.path.dirname(
        os.path.realpath(__file__)), '.complete')
    # generate match set files.
    if os.path.exists(lock_path) and not regenerate:
        print(Notify.INFO, 'Lock file exists without regeneration request. Skip the preparation.', Notify.ENDC)
    else:
        if os.path.exists(lock_path) and not FLAGS.dry_run:
            os.remove(lock_path)
        print(Notify.WARNING, 'Prepare match sets upon request.', Notify.ENDC)
        prog_bar = progressbar.ProgressBar()
        prog_bar.max_value = len(global_info)
        start_time = time.time()
        offset = 0
        for idx, val in enumerate(global_info):
            record = val.split(' ')
            out_match_set_path = os.path.join(
                FLAGS.gl3d, 'data', record[0], 'match_sets')
            in_corr_path = os.path.join(
                FLAGS.gl3d, 'data', record[0], 'geolabel', 'corr.bin')
            kpt_path = os.path.join(FLAGS.gl3d, 'data', record[0], 'img_kpts')
            camera_path = os.path.join(
                FLAGS.gl3d, 'data', record[0], 'geolabel', 'cameras.txt')
            parse_corr_to_match_set(in_corr_path, kpt_path, camera_path, out_match_set_path,
                                    FLAGS.num_corr, offset, dry_run=FLAGS.dry_run,
                                    visualize=False, global_img_list=global_img_list)
            offset = int(record[2])
            prog_bar.update(idx)
        assert offset == len(global_img_list), Notify.FAIL + \
            ' Assertion fails.' + Notify.ENDC
        # create a lock file in case of incomplete preperation.
        open(lock_path, 'w')
        format_str = ('Time cost preparing match sets %.3f sec')
        print(Notify.INFO, format_str %
              (time.time() - start_time), Notify.ENDC)
    # get the match set list.
    imageset_list_name = 'imageset_train.txt' if is_training else 'imageset_test.txt'
    match_set_list = get_match_set_list(os.path.join(
        gl3d_list_folder, imageset_list_name), q_diff_thld, rot_diff_thld)
    return match_set_list, global_img_list, global_depth_list, global_reg_feat_list
