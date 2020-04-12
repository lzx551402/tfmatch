#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Training script for local features.
"""

import os
import time
import yaml

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from model import training
from preprocess import prepare_match_sets
from utils.npy_utils import EndPoints

from tools.common import Notify
from template.misc import summarizer
from template.solver import solver
from template.recoverer import recoverer

FLAGS = tf.app.flags.FLAGS

# Params for config.
tf.app.flags.DEFINE_string('save_dir', None,
                           """Path to save the model.""")
tf.app.flags.DEFINE_string('gl3d', None,
                           """Path to dataset root.""")
tf.app.flags.DEFINE_integer('num_corr', 1024,
                            """The correspondence number of one sample.""")
# Training config
tf.app.flags.DEFINE_string('train_config', None,
                           """Path to training configuration file.""")
tf.app.flags.DEFINE_string('data_split', 'comb',
                           """Which data split in GL3D will be used.""")
tf.app.flags.DEFINE_boolean('is_training', None,
                            """Flag to training model.""")
tf.app.flags.DEFINE_boolean('regenerate', False,
                            """Flag to re-generate training samples.""")
tf.app.flags.DEFINE_boolean('dry_run', False,
                            """Whether to enable dry-run mode in data generation (useful for debugging).""")
tf.app.flags.DEFINE_integer('device_idx', 0,
                            """GPU device index.""")


def train(sample_list, img_list, depth_list, reg_feat_list, train_config):
    """The training procedure.
    Args:
        sample_list: List of training sample file paths.
        img_list: List of image paths.
        depth_list: List of depth paths.
        reg_feat_list: List of regional features.
    Returns:
        Nothing.
    """
    # Construct training networks.
    print(Notify.INFO, 'Running on GPU indexed', FLAGS.device_idx, Notify.ENDC)
    print(Notify.INFO, 'Construct training networks.', Notify.ENDC)
    endpoints = training(sample_list, img_list, depth_list, reg_feat_list, train_config['network'])
    endpoints = EndPoints(endpoints, train_config['network'])
    tot_loss = endpoints.get_total_loss()
    endpoints.disp_terms['Total loss'] = tot_loss
    print_list = endpoints.get_print_list()
    # Construct the solver.
    opt, lr_op, assign_global_step = solver(train_config['solver'], tot_loss)
    summary_op = summarizer(
        {'loss': tot_loss, 'lr': lr_op}, disp_filter='conv0')
    # Create a saver.
    saver = tf.compat.v1.train.Saver(
        tf.compat.v1.global_variables(), max_to_keep=None)
    # Create the initializier.
    init_op = tf.compat.v1.global_variables_initializer()
    # GPU usage grows incrementally.
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    with tf.compat.v1.Session(config=config) as sess:
        dump_config = train_config['dump']
        # Initialize variables.
        print(Notify.INFO, 'Running initialization operator.', Notify.ENDC)
        sess.run(init_op)
        step = recoverer(train_config['recoverer'], sess)
        if train_config['solver']['extra']['global_step'] >= 0:
            step = train_config['solver']['extra']['global_step']
        # Create summary writer.
        if FLAGS.is_training:
            if assign_global_step is not None:
                _ = sess.run([assign_global_step])
            if dump_config['log_dir'] is not None:
                print(Notify.INFO, 'Create log writer.', Notify.ENDC)
                summary_writer = tf.summary.FileWriter(
                    dump_config['log_dir'], sess.graph)
        # Start populating the queue.
        start_time = time.time()
        while step <= dump_config['max_steps']:
            disp_list = []
            returns = sess.run([summary_op, opt, lr_op] + print_list + disp_list)
            summary_str = returns[0]
            out_lr = returns[2]
            out_print_items = returns[3:3 + len(print_list)]
            endpoints.add_print_items(out_print_items)
            # Print info.
            if FLAGS.is_training and step % dump_config['display'] == 0:
                duration = time.time() - start_time
                start_time = time.time()
                format_str = 'Step %d, lr = %.6f (%.3f sec/step)'
                print(Notify.INFO, format_str % (step, out_lr, duration / dump_config['display']),
                      Notify.ENDC)
                endpoints.disp_and_clear()
            # Write summary.
            if FLAGS.is_training and step % (dump_config['display'] * 2) == 0 and dump_config['log_dir'] is not None:
                summary_writer.add_summary(summary_str, step)
            # Save the model checkpoint periodically.
            if FLAGS.is_training and (step % dump_config['snapshot'] == 0 or step == dump_config['max_steps']) and step != 0:
                checkpoint_path = os.path.join(FLAGS.save_dir, 'model.ckpt')
                if step != train_config['recoverer']['ckpt_step']:
                    print(Notify.INFO, 'Save model',
                          checkpoint_path, Notify.ENDC)
                    saver.save(sess, checkpoint_path, global_step=step, write_meta_graph=False)
            step += 1


def main(argv=None):  # pylint: disable=unused-argument
    """Program entrance."""
    with open(FLAGS.train_config, 'r') as f:
        train_config = yaml.load(f, Loader=yaml.FullLoader)
    # Prepare training samples.
    sample_list, img_list, depth_list, reg_feat_list = prepare_match_sets(
        regenerate=FLAGS.regenerate, is_training=FLAGS.is_training, data_split=FLAGS.data_split)
    # Training entrance.
    train(sample_list, img_list, depth_list, reg_feat_list, train_config)


if __name__ == '__main__':
    tf.flags.mark_flags_as_required(['is_training', 'gl3d', 'train_config'])
    if FLAGS.is_training:
        tf.flags.mark_flags_as_required(['save_dir'])
    tf.compat.v1.app.run()
