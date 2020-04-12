"""
Heavily adapted from D2-Net:
https://github.com/mihaidusmanu/d2-net
"""

import tensorflow as tf


def rnd_sample(inputs, n_sample, seed=None):
    cur_size = tf.shape(inputs[0])[0]
    rnd_idx = tf.random.shuffle(tf.range(cur_size), seed=seed)[0:n_sample]
    outputs = [tf.gather(i, rnd_idx) for i in inputs]
    return outputs


def get_dist_mat(feat1, feat2, dist_type):
    eps = 1e-6
    cos_dist_mat = tf.matmul(feat1, feat2, transpose_b=True)
    if dist_type == 'cosine_dist':
        dist_mat = tf.clip_by_value(cos_dist_mat, -1, 1)
    elif dist_type == 'euclidean_dist':
        dist_mat = tf.sqrt(tf.maximum(2 - 2 * cos_dist_mat, eps))
    elif dist_type == 'euclidean_dist_no_norm':
        norm1 = tf.reduce_sum(feat1 * feat1, axis=-1, keepdims=True)
        norm2 = tf.reduce_sum(feat2 * feat2, axis=-1, keepdims=True)
        dist_mat = tf.sqrt(tf.maximum(0., norm1 - 2 * cos_dist_mat +
                                      tf.linalg.matrix_transpose(norm2)) + eps)
    else:
        raise NotImplementedError()
    return dist_mat


def interpolate(pos, inputs, batched=True, nd=True):
    if not batched:
        pos = tf.expand_dims(pos, 0)
        inputs = tf.expand_dims(inputs, 0)

    h = tf.shape(inputs)[1]
    w = tf.shape(inputs)[2]

    i = pos[:, :, 0]
    j = pos[:, :, 1]

    i_top_left = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_top_right = tf.clip_by_value(tf.cast(tf.math.floor(i), tf.int32), 0, h - 1)
    j_top_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    i_bottom_left = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_left = tf.clip_by_value(tf.cast(tf.math.floor(j), tf.int32), 0, w - 1)

    i_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(i), tf.int32), 0, h - 1)
    j_bottom_right = tf.clip_by_value(tf.cast(tf.math.ceil(j), tf.int32), 0, w - 1)

    dist_i_top_left = i - tf.cast(i_top_left, tf.float32)
    dist_j_top_left = j - tf.cast(j_top_left, tf.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1), batch_dims=1) +
        w_top_right * tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1), batch_dims=1) +
        w_bottom_left * tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1), batch_dims=1) +
        w_bottom_right *
        tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1), batch_dims=1)
    )

    if not batched:
        interpolated_val = tf.squeeze(interpolated_val, axis=0)
    return interpolated_val


def validate_and_interpolate(pos, inputs, validate_corner=True, validate_val=None, nd=False):
    if nd:
        h, w, c = inputs.get_shape().as_list()
    else:
        h, w = inputs.get_shape().as_list()
    ids = tf.range(0, tf.shape(pos)[0])

    i = pos[:, 0]
    j = pos[:, 1]

    i_top_left = tf.cast(tf.math.floor(i), tf.int32)
    j_top_left = tf.cast(tf.math.floor(j), tf.int32)

    i_top_right = tf.cast(tf.math.floor(i), tf.int32)
    j_top_right = tf.cast(tf.math.ceil(j), tf.int32)

    i_bottom_left = tf.cast(tf.math.ceil(i), tf.int32)
    j_bottom_left = tf.cast(tf.math.floor(j), tf.int32)

    i_bottom_right = tf.cast(tf.math.ceil(i), tf.int32)
    j_bottom_right = tf.cast(tf.math.ceil(j), tf.int32)

    if validate_corner:
        # Valid corner
        valid_top_left = tf.logical_and(i_top_left >= 0, j_top_left >= 0)
        valid_top_right = tf.logical_and(i_top_right >= 0, j_top_right < w)
        valid_bottom_left = tf.logical_and(i_bottom_left < h, j_bottom_left >= 0)
        valid_bottom_right = tf.logical_and(i_bottom_right < h, j_bottom_right < w)

        valid_corner = tf.logical_and(
            tf.logical_and(valid_top_left, valid_top_right),
            tf.logical_and(valid_bottom_left, valid_bottom_right)
        )

        i_top_left = i_top_left[valid_corner]
        j_top_left = j_top_left[valid_corner]

        i_top_right = i_top_right[valid_corner]
        j_top_right = j_top_right[valid_corner]

        i_bottom_left = i_bottom_left[valid_corner]
        j_bottom_left = j_bottom_left[valid_corner]

        i_bottom_right = i_bottom_right[valid_corner]
        j_bottom_right = j_bottom_right[valid_corner]

        ids = ids[valid_corner]

    if validate_val is not None:
        # Valid depth
        valid_depth = tf.logical_and(
            tf.logical_and(
                tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1)) > 0,
                tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1)) > 0
            ),
            tf.logical_and(
                tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1)) > 0,
                tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1)) > 0
            )
        )

        i_top_left = i_top_left[valid_depth]
        j_top_left = j_top_left[valid_depth]

        i_top_right = i_top_right[valid_depth]
        j_top_right = j_top_right[valid_depth]

        i_bottom_left = i_bottom_left[valid_depth]
        j_bottom_left = j_bottom_left[valid_depth]

        i_bottom_right = i_bottom_right[valid_depth]
        j_bottom_right = j_bottom_right[valid_depth]

        ids = ids[valid_depth]

    # Interpolation
    i = tf.gather(i, ids)
    j = tf.gather(j, ids)
    dist_i_top_left = i - tf.cast(i_top_left, tf.float32)
    dist_j_top_left = j - tf.cast(j_top_left, tf.float32)
    w_top_left = (1 - dist_i_top_left) * (1 - dist_j_top_left)
    w_top_right = (1 - dist_i_top_left) * dist_j_top_left
    w_bottom_left = dist_i_top_left * (1 - dist_j_top_left)
    w_bottom_right = dist_i_top_left * dist_j_top_left

    if nd:
        w_top_left = w_top_left[..., None]
        w_top_right = w_top_right[..., None]
        w_bottom_left = w_bottom_left[..., None]
        w_bottom_right = w_bottom_right[..., None]

    interpolated_val = (
        w_top_left * tf.gather_nd(inputs, tf.stack([i_top_left, j_top_left], axis=-1)) +
        w_top_right * tf.gather_nd(inputs, tf.stack([i_top_right, j_top_right], axis=-1)) +
        w_bottom_left * tf.gather_nd(inputs, tf.stack([i_bottom_left, j_bottom_left], axis=-1)) +
        w_bottom_right * tf.gather_nd(inputs, tf.stack([i_bottom_right, j_bottom_right], axis=-1))
    )

    pos = tf.stack([i, j], axis=1)
    return [interpolated_val, pos, ids]


def get_warp(pos0, rel_pose, depth0, K0, depth1, K1, bs):
    def swap_axis(data):
        return tf.stack([data[:, 1], data[:, 0]], axis=-1)

    all_pos0 = []
    all_pos1 = []
    all_ids = []
    for i in range(bs):
        z0, new_pos0, ids = validate_and_interpolate(pos0[i], depth0[i], validate_val=0)

        uv0_homo = tf.concat([swap_axis(new_pos0), tf.ones((tf.shape(new_pos0)[0], 1))], axis=-1)
        xy0_homo = tf.matmul(tf.linalg.inv(K0[i]), uv0_homo, transpose_b=True)
        xyz0_homo = tf.concat([tf.expand_dims(z0, 0) * xy0_homo,
                               tf.ones((1, tf.shape(new_pos0)[0]))], axis=0)

        xyz1 = tf.matmul(rel_pose[i], xyz0_homo)
        xy1_homo = xyz1 / tf.expand_dims(xyz1[-1, :], axis=0)
        uv1 = tf.transpose(tf.matmul(K1[i], xy1_homo))[:, 0:2]

        new_pos1 = swap_axis(uv1)
        annotated_depth, new_pos1, new_ids = validate_and_interpolate(
            new_pos1, depth1[i], validate_val=0)

        ids = tf.gather(ids, new_ids)
        new_pos0 = tf.gather(new_pos0, new_ids)
        estimated_depth = tf.gather(tf.transpose(xyz1), new_ids)[:, -1]

        inlier_mask = tf.abs(estimated_depth - annotated_depth) < 0.05

        all_ids.append(ids[inlier_mask])
        all_pos0.append(new_pos0[inlier_mask])
        all_pos1.append(new_pos1[inlier_mask])
    return all_pos0, all_pos1, all_ids
