#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
The script defining solvers.
"""

import tensorflow as tf
from tools.common import Notify


def solver(config, sample_loss):
    """Create the solver tensor.
    Args:
        config: Configuration dictionary.
        sample_loss: Loss for a training batch.
    Returns:
        opt: Optimizer tensor.
        lr_op: Learning rate operator.
        global_step: Global step counter.
    """
    if config['extra']['update_var_scope'] is not None:
        update_var_scopes = config['extra']['update_var_scope'].split(',')
        weight_list = []
        bn_list = []
        reg_loss = 0
        for tmp_scope in update_var_scopes:
            weight_list.extend(tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, tmp_scope))
            # Get regularization terms.
            tmp_reg_loss = tf.compat.v1.losses.get_regularization_losses(
                tmp_scope)
            if len(tmp_reg_loss) > 0:
                reg_loss = config['regularization']['weight_decay'] * \
                    tf.add_n(tmp_reg_loss)
            # For networks with batch normalization layers, it is necessary to
            # explicitly fetch their moving statistics and add them to the optimizer.
            bn_list.extend(tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS, tmp_scope))
        print(Notify.WARNING, 'Only update parameters in scope',
              config['extra']['update_var_scope'],
              'including', len(weight_list), 'variables', weight_list, Notify.ENDC)
    else:
        # Get weight variable list.
        weight_list = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES)
        print(Notify.WARNING, 'Update variables', weight_list, Notify.ENDC)
        # Get regularization terms.
        reg_loss = config['regularization']['weight_decay'] \
            * tf.add_n(tf.compat.v1.losses.get_regularization_losses())
        # For networks with batch normalization layers, it is necessary to
        # explicitly fetch their moving statistics and add them to the optimizer.
        bn_list = tf.compat.v1.get_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS)

    # Adjust learning rate in a specific scope.
    if 'adjust_lr' in config['extra']:
        all_adjust_list = []
        all_adjust_lr = []
        for i in config['extra']['adjust_lr']:
            adjust_scope, adjust_lr = i.split(',')
            adjust_lr = float(adjust_lr)
            adjust_list = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, adjust_scope)
            all_adjust_list.append(adjust_list)
            all_adjust_lr.append(adjust_lr)
            print(Notify.WARNING, 'Adjust variables in scope', adjust_scope,
                  'by', adjust_lr, 'of', adjust_list, Notify.ENDC)
            for tmp_weight in adjust_list:
                if tmp_weight in weight_list:
                    weight_list.remove(tmp_weight)
    else:
        all_adjust_list = []
        all_adjust_lr = []

    # Get global step counter.
    global_step = tf.Variable(0, trainable=False, name='global_step')
    if config['extra']['global_step'] >= 0:
        print(Notify.WARNING, 'Reset the global step to',
              config['extra']['global_step'], Notify.ENDC)
        assign_global_step = tf.compat.v1.assign(
            global_step, config['extra']['global_step'])
    else:
        assign_global_step = None

    if config['lr']['policy'] == 'exp':
        # Decay the learning rate exponentially based on step number.
        lr_op = tf.compat.v1.train.exponential_decay(config['lr']['base_lr'],
                                                     global_step=global_step,
                                                     decay_steps=config['lr']['stepvalue'],
                                                     decay_rate=config['lr']['gamma'],
                                                     name='lr')
    elif config['lr']['policy'] == 'const':
        lr_op = tf.constant(config['lr']['base_lr'])
    else:
        raise NotImplementedError()

    with tf.control_dependencies(bn_list):
        # Choose an optimizer and feed the loss.
        if config['optimizer']['name'] == 'Adam':
            opt = tf.compat.v1.train.AdamOptimizer(learning_rate=lr_op).minimize(
                sample_loss + reg_loss, global_step=global_step, var_list=weight_list)
            for i in range(len(all_adjust_list)):
                if all_adjust_list[i]:
                    opt_adjust = tf.compat.v1.train.AdamOptimizer(lr_op * all_adjust_lr[i]).minimize(
                        sample_loss + reg_loss, global_step=global_step, var_list=all_adjust_list[i])
                    opt = tf.group(opt, opt_adjust)
        elif config['optimizer']['name'] == 'SGD':
            opt = tf.compat.v1.train.MomentumOptimizer(lr_op, config['optimizer']['momentum']).minimize(
                sample_loss + reg_loss, global_step=global_step, var_list=weight_list)
            for i in range(len(all_adjust_list)):
                if all_adjust_list[i]:
                    opt_adjust = tf.compat.v1.train.MomentumOptimizer(lr_op * all_adjust_lr[i], config['optimizer']['momentum']).minimize(
                        sample_loss + reg_loss, global_step=global_step, var_list=all_adjust_list[i])
                    opt = tf.group(opt, opt_adjust)
        else:
            raise NotImplementedError(
                'Illegal optimizer type ' + config['optimizer']['name'])
    return opt, lr_op, assign_global_step
