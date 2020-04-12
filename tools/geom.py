#!/usr/bin/env python3
"""
Copyright 2018, Zixin Luo, HKUST.
Geometry computations.
"""

import numpy as np
import cv2
import math


def evaluate_R_t(R_gt, t_gt, R, t, eps=1e-15):
    t = t.flatten()
    t_gt = t_gt.flatten()

    q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    return err_q, err_t


def quaternion_from_matrix(matrix, isprecise=False):
    """Return quaternion from rotation matrix.
    If isprecise is True, the input matrix is assumed to be a precise rotation
    matrix and a faster algorithm is used.
    >>> q = quaternion_from_matrix(numpy.identity(4), True)
    >>> numpy.allclose(q, [1, 0, 0, 0])
    True
    >>> q = quaternion_from_matrix(numpy.diag([1, -1, -1, 1]))
    >>> numpy.allclose(q, [0, 1, 0, 0]) or numpy.allclose(q, [0, -1, 0, 0])
    True
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R, True)
    >>> numpy.allclose(q, [0.9981095, 0.0164262, 0.0328524, 0.0492786])
    True
    >>> R = [[-0.545, 0.797, 0.260, 0], [0.733, 0.603, -0.313, 0],
    ...      [-0.407, 0.021, -0.913, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.19069, 0.43736, 0.87485, -0.083611])
    True
    >>> R = [[0.395, 0.362, 0.843, 0], [-0.626, 0.796, -0.056, 0],
    ...      [-0.677, -0.498, 0.529, 0], [0, 0, 0, 1]]
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.82336615, -0.13610694, 0.46344705, -0.29792603])
    True
    >>> R = random_rotation_matrix()
    >>> q = quaternion_from_matrix(R)
    >>> is_same_transform(R, quaternion_matrix(q))
    True
    >>> R = euler_matrix(0.0, 0.0, numpy.pi/2.0)
    >>> numpy.allclose(quaternion_from_matrix(R, isprecise=False),
    ...                quaternion_from_matrix(R, isprecise=True))
    True
    """
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    if isprecise:
        q = np.empty((4, ))
        t = np.trace(M)
        if t > M[3, 3]:
            q[0] = t
            q[3] = M[1, 0] - M[0, 1]
            q[2] = M[0, 2] - M[2, 0]
            q[1] = M[2, 1] - M[1, 2]
        else:
            i, j, k = 1, 2, 3
            if M[1, 1] > M[0, 0]:
                i, j, k = 2, 3, 1
            if M[2, 2] > M[i, i]:
                i, j, k = 3, 1, 2
            t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
            q[i] = t
            q[j] = M[i, j] + M[j, i]
            q[k] = M[k, i] + M[i, k]
            q[3] = M[k, j] - M[j, k]
        q *= 0.5 / math.sqrt(t * M[3, 3])
    else:
        m00 = M[0, 0]
        m01 = M[0, 1]
        m02 = M[0, 2]
        m10 = M[1, 0]
        m11 = M[1, 1]
        m12 = M[1, 2]
        m20 = M[2, 0]
        m21 = M[2, 1]
        m22 = M[2, 2]
        # symmetric matrix K
        K = np.array([[m00-m11-m22, 0.0,         0.0,         0.0],
                      [m01+m10,     m11-m00-m22, 0.0,         0.0],
                      [m02+m20,     m12+m21,     m22-m00-m11, 0.0],
                      [m21-m12,     m02-m20,     m10-m01,     m00+m11+m22]])
        K /= 3.0
        # quaternion is eigenvector of K that corresponds to largest eigenvalue
        w, V = np.linalg.eigh(K)
        q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        np.negative(q, q)
    return q


def get_relative_pose(pose0, pose1):
    """Compute relative pose.
    Args:
        pose: [R, t]
    Returns:
        rel_pose: [rel_R, rel_t]
    """
    rel_R = np.matmul(pose1[0], pose0[0].T)
    center0 = -np.matmul(pose0[1].T, pose0[0]).T
    center1 = -np.matmul(pose1[1].T, pose1[0]).T
    rel_t = np.matmul(pose1[0], center0 - center1)
    return [rel_R, rel_t]


def skew_symmetric_mat(v):
    v = v.flatten()
    M = np.stack([
        (0, -v[2], v[1]),
        (v[2], 0, -v[0]),
        (-v[1], v[0], 0),
    ], axis=0)
    return M


def get_essential_mat(t0, t1, R0, R1):
    """
    Args:
        t: 3x1 mat.
        R: 3x3 mat.
    Returns:
        e_mat: 3x3 essential matrix.
    """
    dR = np.matmul(R1, R0.T)  # dR = R_1 * R_0^T
    dt = t1 - np.matmul(dR, t0)  # dt = t_1 - dR * t_0

    dt = dt.reshape(1, 3)
    dt_ssm = skew_symmetric_mat(dt)

    e_mat = np.matmul(dt_ssm, dR)  # E = dt_ssm * dR
    e_mat = e_mat / np.linalg.norm(e_mat)
    return e_mat


def undist_points(pts, K, dist, img_size=None):
    n = pts.shape[0]
    new_pts = pts
    if img_size is not None:
        hs = img_size / 2
        new_pts = np.stack([pts[:, 2] * hs[0] + hs[0], pts[:, 5] * hs[1] + hs[1]], axis=1)

    new_dist = np.zeros((5), dtype=np.float32)
    new_dist[0] = dist[0]
    new_dist[1] = dist[1]
    new_dist[4] = dist[2]

    upts = cv2.undistortPoints(np.expand_dims(new_pts, axis=1), K, new_dist)
    upts = np.squeeze(cv2.convertPointsToHomogeneous(upts), axis=1)
    upts = np.matmul(K, upts.T).T[:, 0:2]

    if img_size is not None:
        new_upts = pts.copy()
        new_upts[:, 2] = (upts[:, 0] - hs[0]) / hs[0]
        new_upts[:, 5] = (upts[:, 1] - hs[1]) / hs[1]
        return new_upts
    else:
        return upts
