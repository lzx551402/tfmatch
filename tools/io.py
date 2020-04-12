#!/usr/bin/env python3
"""
Copyright 2017, Zixin Luo, HKUST.
IO tools.
"""

import os
import re
import xml.etree.ElementTree
from math import isnan
from struct import pack, unpack
from random import shuffle, sample

import cv2
import numpy as np
from tools.common import Notify
from tools.geom import get_essential_mat, undist_points, get_relative_pose, evaluate_R_t


def draw_kpts(imgs, kpts, color=(0, 255, 0), radius=2, thickness=2):
    """
    Args:
        imgs: color images.
        kpts: Nx2 numpy array.
    Returns:
        all_display: image with drawn keypoints.
    """
    all_display = []
    for idx, val in enumerate(imgs):
        kpt = kpts[idx]
        tmp_img = val.copy()
        for kpt_idx in range(kpt.shape[0]):
            display = cv2.circle(
                tmp_img, (int(kpt[kpt_idx][0]), int(kpt[kpt_idx][1])), radius, color, thickness)
        all_display.append(display)
    all_display = np.concatenate(all_display, axis=1)
    return all_display


def load_pfm(pfm_path):
    with open(pfm_path, 'rb') as fin:
        color = None
        width = None
        height = None
        scale = None
        data_type = None
        header = str(fin.readline().decode('UTF-8')).rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')

        dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                             fin.readline().decode('UTF-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')
        scale = float((fin.readline().decode('UTF-8')).rstrip())
        if scale < 0:  # little-endian
            data_type = '<f'
        else:
            data_type = '>f'  # big-endian
        data_string = fin.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.flip(data, 0)
    return data


def read_list(list_path):
    """Read list."""
    if list_path is None or not os.path.exists(list_path):
        print(Notify.FAIL, 'Not exist', list_path, Notify.ENDC)
        exit(-1)
    content = open(list_path).read().splitlines()
    return content


def read_cams(cam_path):
    """
    Args:
        cam_path: Path to cameras.txt.
    Returns:
        cam_dict: A dictionary indexed by image index and composed of (K, t, R, dist, img_size).
        K - 2x3, t - 3x1, R - 3x3, dist - 1x3, img_size - 1x2.
    """
    cam_data = [i.split(' ') for i in read_list(cam_path)]

    cam_dict = {}
    for i in cam_data:
        i = [float(j) for j in i if j is not '']
        K = np.array([(i[1], i[5], i[3]),
                      (0, i[2], i[4]), (0, 0, 1)])
        t = np.array([(i[6], ), (i[7], ), (i[8], )])
        R = np.array([(i[9], i[10], i[11]),
                      (i[12], i[13], i[14]),
                      (i[15], i[16], i[17])])
        dist = np.array([i[18], i[19], i[20]])
        img_size = np.array([i[21], i[22]])
        cam_dict[i[0]] = (K, t, R, dist, img_size)
    return cam_dict


def read_corr(file_path):
    """Read local match file.
    Args:
        file_path: correspondence file path.
    Returns:
        matches: List of match data, each consists of two image indices and Nx15 match matrix.
        6 affine params for image 1 + 6 affine params for image 2 + geo_dist + feat_idx 1 + feat_idx 2
    """
    matches = []
    with open(file_path, 'rb') as fin:
        while True:
            rin = fin.read(24)
            if len(rin) == 0:
                # EOF
                break
            idx0, idx1, num = unpack('L' * 3, rin)
            # bytes_theta = num * 52
            bytes_theta = num * 60
            corr = np.fromstring(fin.read(bytes_theta),
                                 dtype=np.float32).reshape(-1, 15)
            matches.append([idx0, idx1, corr])
    return matches


def parse_corr_to_match_set(input_corr, kpt_path, camera_path, out_root, match_set_size, offset,
                            min_inlier_ratio=0.01, visualize=False, thld_q=3, dry_run=False, global_img_list=None):
    """Parse the correspondence file to multiple training match sets."""
    if not os.path.exists(out_root):
        os.mkdir(out_root)
    else:
        if not dry_run:
            _ = [os.remove(os.path.join(out_root, tmp_sample))
                 for tmp_sample in os.listdir(out_root)]
    matches = read_corr(input_corr)
    cam_dict = read_cams(camera_path)
    # maintain a dictionary to hold loaded keypoint files.
    kpt_dict = {}
    for idx, val in enumerate(matches):
        cidx = (val[0], val[1])
        corr = val[2]
        img_idx = (val[0] + offset, val[1] + offset)
        num_inlier = corr.shape[0]
        # measure the rotation difference
        t0, t1 = cam_dict[cidx[0]][1], cam_dict[cidx[1]][1]
        R0, R1 = cam_dict[cidx[0]][2], cam_dict[cidx[1]][2]

        err_q, _ = evaluate_R_t(R0, t0, R1, t1)
        err_q = int(err_q / 3.14 * 180)

        K = (cam_dict[cidx[0]][0], cam_dict[cidx[1]][0])
        dist = (cam_dict[cidx[0]][3], cam_dict[cidx[1]][3])
        ori_img_size = (cam_dict[cidx[0]][4], cam_dict[cidx[1]][4])

        if err_q < thld_q:
            continue
        # read kpt file.
        for z in range(2):
            # avoid reading the same keypoint file for multiple times.
            if img_idx[z] not in kpt_dict:
                tmp_kpt_path = os.path.join(
                    kpt_path, str(val[z]).zfill(8) + '.bin')
                kpt_data = np.fromfile(tmp_kpt_path, dtype=np.float32)
                kpt_data = np.reshape(kpt_data, (-1, 6))
                kpt_data = undist_points(
                    kpt_data, K[z], dist[z], img_size=ori_img_size[z])
                kpt_dict[img_idx[z]] = kpt_data
        # fetch the keypoint files for current match.
        all_kpt0 = kpt_dict[img_idx[0]]
        all_kpt1 = kpt_dict[img_idx[1]]
        if all_kpt0.shape[0] < match_set_size or all_kpt1.shape[0] < match_set_size:
            continue
        # get indices of inlier matches.
        inlier_ratio = float(num_inlier) / match_set_size
        if inlier_ratio < min_inlier_ratio:
            continue
        elif inlier_ratio > 1.0:
            inlier_idx = sample(range(num_inlier), match_set_size)
        else:
            inlier_idx = range(num_inlier)

        geo_sim = corr[inlier_idx, 12]
        geo_sim = np.array([1. if isnan(i_) else i_ for i_ in geo_sim])
        inlier_feat_idx = corr[inlier_idx, 13:15].astype(np.int32)
        inlier_kpt0 = all_kpt0[inlier_feat_idx[:, 0]]
        inlier_kpt1 = all_kpt1[inlier_feat_idx[:, 1]]

        rot0 = np.arctan2(inlier_kpt0[:, 1], inlier_kpt0[:, 0])
        rot1 = np.arctan2(inlier_kpt1[:, 1], inlier_kpt1[:, 0])
        rot_diff = np.mean(np.abs(rot0 - rot1))
        rot_diff = int(rot_diff / 3.14 * 180)

        num_noise = match_set_size - min(num_inlier, match_set_size)

        # exclude keypoints that establish matches.
        noise_kpt0 = np.delete(
            all_kpt0, inlier_feat_idx[:, 0].astype(np.int32), axis=0)
        noise_kpt1 = np.delete(
            all_kpt1, inlier_feat_idx[:, 1].astype(np.int32), axis=0)

        if num_noise > 0:
            noise_idx0 = sample(range(noise_kpt0.shape[0]), num_noise)
            noise_idx1 = sample(range(noise_kpt1.shape[0]), num_noise)
            noise0 = noise_kpt0[noise_idx0]
            noise1 = noise_kpt1[noise_idx1]

            assert len(noise_idx0) == num_noise
            assert len(noise_idx1) == num_noise

            sampled_kpt0 = np.concatenate((inlier_kpt0, noise0), axis=0)
            sampled_kpt1 = np.concatenate((inlier_kpt1, noise1), axis=0)
            geo_sim = np.concatenate((geo_sim, np.zeros((num_noise,))))
        else:
            sampled_kpt0 = inlier_kpt0
            sampled_kpt1 = inlier_kpt1

        e_mat = get_essential_mat(t0, t1, R0, R1).flatten()
        rel_pose = get_relative_pose([R0, t0], [R1, t1])
        rel_pose = np.concatenate(rel_pose, axis=-1).flatten()

        theta = np.concatenate(
            (sampled_kpt0.flatten(), sampled_kpt1.flatten(), geo_sim))

        # length: 2 + 1 + 2 + 2 + 9 + 9 + 9 + 12 = 46
        geom_data = np.concatenate([np.array(img_idx),
                                    np.expand_dims(
                                        np.array(min(num_inlier, match_set_size)), axis=0),
                                    ori_img_size[0], ori_img_size[1],
                                    K[0].flatten(), K[1].flatten(), e_mat, rel_pose], axis=0)

        if visualize:
            import matplotlib.pyplot as plt
            img0 = cv2.imread(global_img_list[img_idx[0]])[..., ::-1]
            img1 = cv2.imread(global_img_list[img_idx[1]])[..., ::-1]

            tmp_kpts0 = np.stack([all_kpt0[:, 2], all_kpt0[:, 5]], axis=-1)
            tmp_kpts1 = np.stack([all_kpt1[:, 2], all_kpt1[:, 5]], axis=-1)
            tmp_img_size0 = np.array((img0.shape[1], img0.shape[0]))
            tmp_img_size1 = np.array((img1.shape[1], img1.shape[0]))
            tmp_kpts0 = tmp_kpts0 * tmp_img_size0 / 2 + tmp_img_size0 / 2
            tmp_kpts1 = tmp_kpts1 * tmp_img_size1 / 2 + tmp_img_size1 / 2

            display = draw_kpts([img0, img1], [tmp_kpts0, tmp_kpts1])
            cv2.imshow("disp.jpg", display)
            cv2.waitKey()

        match_set_path = os.path.join(out_root, '_'.join(
            [str(idx), str(0), "%d" % err_q, "%d" % rot_diff]) + '.match_set')
        if not dry_run:
            with open(match_set_path, 'wb') as fout:
                fout.write(pack('f' * geom_data.shape[0], *geom_data))
                fout.write(pack('f' * match_set_size * 13, *theta))
