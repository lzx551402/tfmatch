#!/usr/bin/env python3
"""
Copyright 2018, Zixin Luo, HKUST.
The script defining the recoverer.
"""

import tensorflow as tf

from tools.common import Notify


def recoverer(config, sess):
    """
    Recovery parameters from a pretrained model.
    Args:
        config: The recoverer configuration.
        sess: The tensorflow session instance.
    Returns:
        step: The step value.
    """
    if config['pretrained_model'] is not None and config['ckpt_step'] is not None:
        restore_var = []
        # selectively recover the parameters.
        if config['exclude_var'] is None:
            restore_var = tf.compat.v1.global_variables()
        else:
            keyword = config['exclude_var'].split(',')
            for tmp_var in tf.compat.v1.global_variables():
                find_keyword = False
                for tmp_keyword in keyword:
                    if tmp_var.name.find(tmp_keyword) >= 0:
                        print(Notify.WARNING, 'Ignore the recovery of variable',
                              tmp_var.name, Notify.ENDC)
                        find_keyword = True
                        break
                if not find_keyword:
                    restore_var.append(tmp_var)
        restorer = tf.compat.v1.train.Saver(restore_var)
        restorer.restore(sess, config['pretrained_model'])
        print(Notify.INFO, 'Pre-trained model restored from %s' %
              config['pretrained_model'], Notify.ENDC)
        return config['ckpt_step']
    else:
        print(Notify.WARNING, 'Training from scratch.', Notify.ENDC)
        return 0
