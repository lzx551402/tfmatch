#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
Miscellaneous utilities.
"""

import tensorflow as tf


def summarizer(summary_dict, weight_list=None, disp_filter=None):
    if summary_dict is not None:
        for key, val in summary_dict.items():
            tf.compat.v1.summary.scalar(key, val)

    if weight_list is not None:
        for var in weight_list:
            tf.compat.v1.summary.histogram(var.op.name, var)

    # Build the summary operation.
    summary_op = tf.compat.v1.summary.merge_all()
    return summary_op
