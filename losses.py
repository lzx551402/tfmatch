import tensorflow as tf
from utils.tf_geom import get_dist_mat, rnd_sample, interpolate


def make_detector_loss(pos0, pos1, dense_feat_map0, dense_feat_map1,
               score_map0, score_map1, batch_size, num_corr, loss_type, config):
    joint_loss = tf.constant(0.)
    accuracy = tf.constant(0.)
    all_valid_pos0 = []
    all_valid_pos1 = []
    all_valid_match = []
    for i in range(batch_size):
        # random sample
        valid_pos0, valid_pos1 = rnd_sample([pos0[i], pos1[i]], num_corr)
        valid_num = tf.shape(valid_pos0)[0]

        valid_feat0 = interpolate(
            valid_pos0 / 4, dense_feat_map0[i], batched=False)
        valid_feat1 = interpolate(
            valid_pos1 / 4, dense_feat_map1[i], batched=False)

        valid_feat0 = tf.nn.l2_normalize(valid_feat0, axis=-1)
        valid_feat1 = tf.nn.l2_normalize(valid_feat1, axis=-1)

        valid_score0 = interpolate(valid_pos0, tf.squeeze(
            score_map0[i], axis=-1), nd=False, batched=False)
        valid_score1 = interpolate(valid_pos1, tf.squeeze(
            score_map1[i], axis=-1), nd=False, batched=False)
        if config['det']['corr_weight']:
            corr_weight = valid_score0 * valid_score1
        else:
            corr_weight = None

        safe_radius = config['det']['safe_radius']
        if safe_radius > 0:
            radius_mask_row = get_dist_mat(
                valid_pos1, valid_pos1, "euclidean_dist_no_norm")
            radius_mask_row = tf.less(radius_mask_row, safe_radius)
            radius_mask_col = get_dist_mat(
                valid_pos0, valid_pos0, "euclidean_dist_no_norm")
            radius_mask_col = tf.less(radius_mask_col, safe_radius)
            radius_mask_row = tf.cast(
                radius_mask_row, tf.float32) - tf.eye(valid_num)
            radius_mask_col = tf.cast(
                radius_mask_col, tf.float32) - tf.eye(valid_num)
        else:
            radius_mask_row = None
            radius_mask_col = None

        si_loss, si_accuracy, matched_mask = tf.cond(
            tf.less(valid_num, 32),
            lambda: (tf.constant(0.), tf.constant(1.),
                     tf.cast(tf.zeros((1, valid_num)), tf.bool)),
            lambda: make_structured_loss(
                tf.expand_dims(valid_feat0, 0), tf.expand_dims(valid_feat1, 0),
                loss_type=loss_type,
                radius_mask_row=radius_mask_row, radius_mask_col=radius_mask_col,
                corr_weight=tf.expand_dims(corr_weight, 0) if corr_weight is not None else None,
                name='si_loss')
        )

        joint_loss += si_loss / batch_size
        accuracy += si_accuracy / batch_size
        all_valid_match.append(tf.squeeze(matched_mask, axis=0))
        all_valid_pos0.append(valid_pos0)
        all_valid_pos1.append(valid_pos1)

    return joint_loss, accuracy


def make_quadruple_loss(kpt_m0, kpt_m1, inlier_num):
    batch_size = kpt_m0.get_shape()[0].value
    num_corr = kpt_m1.get_shape()[1].value
    kpt_m_diff0 = tf.linalg.matrix_transpose(
        tf.tile(kpt_m0, (1, 1, num_corr))) - kpt_m0
    kpt_m_diff1 = tf.linalg.matrix_transpose(
        tf.tile(kpt_m1, (1, 1, num_corr))) - kpt_m1

    R = kpt_m_diff0 * kpt_m_diff1

    quad_loss = 0
    accuracy = 0
    for i in range(batch_size):
        cur_inlier_num = tf.squeeze(inlier_num[i])
        inlier_block = R[i, 0:cur_inlier_num, 0:cur_inlier_num]
        inlier_block = inlier_block + tf.eye(cur_inlier_num)
        inlier_block = tf.maximum(0., 1. - inlier_block)
        error = tf.count_nonzero(inlier_block)
        cur_inlier_num = tf.cast(cur_inlier_num, tf.float32)
        quad_loss += tf.reduce_sum(inlier_block) / \
            (cur_inlier_num * (cur_inlier_num - 1))
        accuracy += 1. - tf.cast(error, tf.float32) / \
            (cur_inlier_num * (cur_inlier_num - 1))

    quad_loss /= float(batch_size)
    accuracy /= float(batch_size)
    return quad_loss, accuracy


def make_structured_loss(feat_anc, feat_pos,
                         loss_type='RATIO', inlier_mask=None,
                         radius_mask_row=None, radius_mask_col=None,
                         corr_weight=None, dist_mat=None, name='loss'):
    """
    Structured loss construction.
    Args:
        feat_anc, feat_pos: Feature matrix.
        loss_type: Loss type.
        inlier_mask:
    Returns:

    """
    batch_size = feat_anc.get_shape()[0].value
    num_corr = tf.shape(feat_anc)[1]
    if inlier_mask is None:
        inlier_mask = tf.cast(tf.ones((batch_size, num_corr)), tf.bool)
    inlier_num = tf.count_nonzero(tf.cast(inlier_mask, tf.float32), axis=-1)

    if loss_type == 'LOG' or loss_type == 'L2NET' or loss_type == 'CIRCLE':
        dist_type = 'cosine_dist'
    elif loss_type.find('HARD') >= 0:
        dist_type = 'euclidean_dist'
    else:
        raise NotImplementedError()

    if dist_mat is None:
        dist_mat = get_dist_mat(feat_anc, feat_pos, dist_type)
    pos_vec = tf.linalg.diag_part(dist_mat)

    if loss_type.find('HARD') >= 0:
        neg_margin = 1
        dist_mat_without_min_on_diag = dist_mat + \
            10 * tf.expand_dims(tf.eye(num_corr), 0)
        mask = tf.cast(
            tf.less(dist_mat_without_min_on_diag, 0.008), tf.float32)
        dist_mat_without_min_on_diag += mask*10

        if radius_mask_row is not None:
            hard_neg_dist_row = dist_mat_without_min_on_diag + 10 * radius_mask_row
        else:
            hard_neg_dist_row = dist_mat_without_min_on_diag
        if radius_mask_col is not None:
            hard_neg_dist_col = dist_mat_without_min_on_diag + 10 * radius_mask_col
        else:
            hard_neg_dist_col = dist_mat_without_min_on_diag

        hard_neg_dist_row = tf.reduce_min(hard_neg_dist_row, axis=-1)
        hard_neg_dist_col = tf.reduce_min(hard_neg_dist_col, axis=-2)

        if loss_type == 'HARD_TRIPLET':
            loss_row = tf.maximum(neg_margin + pos_vec - hard_neg_dist_row, 0)
            loss_col = tf.maximum(neg_margin + pos_vec - hard_neg_dist_col, 0)
        elif loss_type == 'HARD_CONTRASTIVE':
            pos_margin = 0.2
            pos_loss = tf.maximum(pos_vec - pos_margin, 0)
            loss_row = pos_loss + tf.maximum(neg_margin - hard_neg_dist_row, 0)
            loss_col = pos_loss + tf.maximum(neg_margin - hard_neg_dist_col, 0)
        else:
            raise NotImplementedError()

    elif loss_type == 'LOG' or loss_type == 'L2NET':
        if loss_type == 'LOG':
            with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
                log_scale = tf.compat.v1.get_variable('scale_temperature', shape=(), dtype=tf.float32,
                                                      initializer=tf.constant_initializer(1))
                tf.compat.v1.add_to_collection(
                    tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES, log_scale)
        else:
            log_scale = tf.constant(1.)
        softmax_row = tf.nn.softmax(log_scale * dist_mat, axis=2)
        softmax_col = tf.nn.softmax(log_scale * dist_mat, axis=1)

        loss_row = -tf.math.log(tf.linalg.diag_part(softmax_row))
        loss_col = -tf.math.log(tf.linalg.diag_part(softmax_col))
    
    elif loss_type == 'CIRCLE':
        log_scale = 512
        m = 0.1
        neg_mask_row = tf.expand_dims(tf.eye(num_corr), 0)
        if radius_mask_row is not None:
            neg_mask_row += radius_mask_row
        neg_mask_col = tf.expand_dims(tf.eye(num_corr), 0)
        if radius_mask_col is not None:
            neg_mask_col += radius_mask_col

        pos_margin = 1 - m
        neg_margin = m
        pos_optimal = 1 + m
        neg_optimal = -m

        neg_mat_row = dist_mat - 128 * neg_mask_row
        neg_mat_col = dist_mat - 128 * neg_mask_col

        lse_positive = tf.math.reduce_logsumexp(-log_scale * (pos_vec[..., None] - pos_margin) * \
                    tf.stop_gradient(tf.maximum(pos_optimal - pos_vec[..., None], 0)), axis=-1)
        
        lse_negative_row = tf.math.reduce_logsumexp(log_scale * (neg_mat_row - neg_margin) * \
                    tf.stop_gradient(tf.maximum(neg_mat_row - neg_optimal, 0)), axis=-1)

        lse_negative_col = tf.math.reduce_logsumexp(log_scale * (neg_mat_col - neg_margin) * \
                    tf.stop_gradient(tf.maximum(neg_mat_col - neg_optimal, 0)), axis=-2)

        loss_row = tf.math.softplus(lse_positive + lse_negative_row) / log_scale
        loss_col = tf.math.softplus(lse_positive + lse_negative_col) / log_scale

    else:
        raise NotImplementedError()

    if dist_type == 'cosine_dist':
        err_row = dist_mat - tf.expand_dims(pos_vec, -1)
        err_col = dist_mat - tf.expand_dims(pos_vec, -2)
    elif dist_type == 'euclidean_dist' or dist_type == 'euclidean_dist_no_norm':
        err_row = tf.expand_dims(pos_vec, -1) - dist_mat
        err_col = tf.expand_dims(pos_vec, -2) - dist_mat
    else:
        raise NotImplementedError()
    if radius_mask_row is not None:
        err_row = err_row - 10 * radius_mask_row
    if radius_mask_col is not None:
        err_col = err_col - 10 * radius_mask_col
    err_row = tf.reduce_sum(tf.maximum(err_row, 0), axis=-1)
    err_col = tf.reduce_sum(tf.maximum(err_col, 0), axis=-2)

    loss = 0
    accuracy = 0

    tot_loss = (loss_row + loss_col) / 2
    if corr_weight is not None:
        tot_loss = tot_loss * corr_weight

    for i in range(batch_size):
        if corr_weight is not None:
            loss += tf.reduce_sum(tot_loss[i][inlier_mask[i]]) / \
                (tf.reduce_sum(corr_weight[i][inlier_mask[i]]) + 1e-6)
        else:
            loss += tf.reduce_mean(tot_loss[i][inlier_mask[i]])
        cnt_err_row = tf.count_nonzero(
            err_row[i][inlier_mask[i]], dtype=tf.float32)
        cnt_err_col = tf.count_nonzero(
            err_col[i][inlier_mask[i]], dtype=tf.float32)
        tot_err = cnt_err_row + cnt_err_col
        accuracy += 1. - \
            tf.math.divide_no_nan(tot_err, tf.cast(
                inlier_num[i], tf.float32)) / batch_size / 2.

    matched_mask = tf.logical_and(tf.equal(err_row, 0), tf.equal(err_col, 0))
    matched_mask = tf.logical_and(matched_mask, inlier_mask)

    loss /= batch_size
    accuracy /= batch_size

    return loss, accuracy, matched_mask
