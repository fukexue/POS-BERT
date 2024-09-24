import os
import glob
import h5py
import math
import torch
import numpy as np
from typing import Dict, List
from torch.utils.data import Dataset
from torchvision import transforms

from scipy.spatial.transform import Rotation
from scipy.stats import special_ortho_group
from scipy import spatial as spt
import utils_3d.se3 as se3
import utils_3d.so3 as so3
from utils_3d.random import uniform_2_sphere

import lmdb
import msgpack_numpy


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    all_data = []
    all_label = []
    for h5_name in sorted(glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition))):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class SaveRawPoint:
    """Clones the point cloud into separate source and reference point clouds"""
    def __call__(self, sample: Dict):
        if isinstance(sample['points_raw'], torch.Tensor):
            sample['points'] = sample['points_raw'].detach()
        else:  # is numpy
            sample['points'] = sample['points_raw'].copy()
        return sample


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample:
            np.random.seed(sample['idx'])

        sample['points'] = self._resample(sample['points'], self.num)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.
        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        elif points.shape[0] == k:
            return points
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class Resampler_firstn:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = sample['points'][:self.num]
        return sample


class RandomTransformSE3:
    def __init__(self, rot_mag: float = 180.0, trans_mag: float = 1.0, random_mag: bool = False):
        """Applies a random rigid transformation to the source point cloud

        Args:
            rot_mag (float): Maximum rotation in degrees
            trans_mag (float): Maximum translation T. Random translation will
              be in the range [-X,X] in each axis
            random_mag (bool): If true, will randomize the maximum rotation, i.e. will bias towards small
                               perturbations
        """
        self._rot_mag = rot_mag
        self._trans_mag = trans_mag
        self._random_mag = random_mag

    def generate_transform(self):
        """Generate a random SE3 transformation (3, 4) """

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        rand_rot = special_ortho_group.rvs(3)
        axis_angle = Rotation.as_rotvec(Rotation.from_dcm(rand_rot))
        axis_angle *= rot_mag / 180.0
        rand_rot = Rotation.from_rotvec(axis_angle).as_dcm()

        # Generate translation
        rand_trans = np.random.uniform(-trans_mag, trans_mag, 3)
        rand_SE3 = np.concatenate((rand_rot, rand_trans[:, None]), axis=1).astype(np.float32)

        return rand_SE3

    def apply_transform(self, p0, transform_mat):
        p1 = se3.transform(transform_mat, p0[:, :3])
        if p0.shape[1] == 6:  # Need to rotate normals also
            n1 = so3.transform(transform_mat[:3, :3], p0[:, 3:6])
            p1 = np.concatenate((p1, n1), axis=-1)

        igt = transform_mat
        gt = se3.inverse(igt)

        return p1, gt, igt

    def transform(self, tensor):
        transform_mat = self.generate_transform()
        return self.apply_transform(tensor, transform_mat)

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])

        sample['points'], transform_r_s, transform_s_r = self.transform(sample['points'])
        # transform_r_s Apply to source to get reference

        return sample


# noinspection PyPep8Naming
class RandomTransformSE3_euler(RandomTransformSE3):
    """Same as RandomTransformSE3, but rotates using euler angle rotations

    This transformation is consistent to Deep Closest Point but does not
    generate uniform rotations

    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            rot_mag, trans_mag = attentuation * self._rot_mag, attentuation * self._trans_mag
        else:
            rot_mag, trans_mag = self._rot_mag, self._trans_mag

        # Generate rotation
        anglex = np.random.uniform() * np.pi * rot_mag / 180.0
        angley = np.random.uniform() * np.pi * rot_mag / 180.0
        anglez = np.random.uniform() * np.pi * rot_mag / 180.0

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                       [0, cosx, -sinx],
                       [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                       [0, 1, 0],
                       [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                       [sinz, cosz, 0],
                       [0, 0, 1]])
        R_ab = Rx @ Ry @ Rz
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomTranslation(RandomTransformSE3):
    """Same as RandomTransformSE3, but no rotations
    """
    def generate_transform(self):

        if self._random_mag:
            attentuation = np.random.random()
            trans_mag = attentuation * self._trans_mag
        else:
            trans_mag = self._trans_mag

        R_ab = np.eye(3)
        t_ab = np.random.uniform(-trans_mag, trans_mag, 3)

        rand_SE3 = np.concatenate((R_ab, t_ab[:, None]), axis=1).astype(np.float32)
        return rand_SE3


class RandomJitter:
    """ generate perturbations """
    def __init__(self, scale=0.01, clip=0.05):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 3)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :3] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = self.jitter(sample['points'])
        return sample


class RandomCrop_dense:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self, p_keep: List = None):
        # p_keep[0] min retained ratio, p_keep[1] max retained ratio
        if p_keep is None:
            p_keep = [0.7, 1.0]  # Crop both clouds to 70%
        self.p_keep = np.array(p_keep, dtype=np.float32)

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
            # 消除裁减后点数不是一样的问题，主要在SHAPENET中存在
            if sum(mask)<int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==False)[0].tolist()[-(int(np.ceil(p_keep*points.shape[0]))-sum(mask)):]]=True
            elif sum(mask)>int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==True)[0].tolist()[-(sum(mask)-int(np.ceil(p_keep*points.shape[0]))):]]=False

        return points[mask, :]

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['crop_proportion'] = self.p_keep
        keep_ratio = np.random.uniform(self.p_keep[0], self.p_keep[1])
        if keep_ratio == 1.0:
            return sample  # No need crop

        sample['points'] = self.crop(sample['points'], keep_ratio)
        return sample


class RandomCrop_Global_dense:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self):
        pass

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
            # 消除裁减后点数不是一样的问题，主要在SHAPENET中存在
            if sum(mask)<int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==False)[0].tolist()[-(int(np.ceil(p_keep*points.shape[0]))-sum(mask)):]]=True
            elif sum(mask)>int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==True)[0].tolist()[-(sum(mask)-int(np.ceil(p_keep*points.shape[0]))):]]=False

        return points[mask, :]

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = self.crop(sample['points'], sample['global_keep_diff_ratio'] if 'global_keep_diff_ratio' in sample else sample['global_keep_unify_ratio'])
        return sample

class RandomCrop_Global_spherical:
    def __init__(self):
        pass

    @staticmethod
    def crop(points, p_keep):
        random_point = points[np.random.choice(points.shape[0], 1, replace=False),:3]
        ckt = spt.cKDTree(points)
        knn_dist, knn_idx = ckt.query(random_point, int(np.ceil(points.shape[0]*p_keep)))
        crop_point = points[knn_idx[0]]
        return crop_point

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = self.crop(sample['points'], sample['global_keep_diff_ratio'] if 'global_keep_diff_ratio' in sample else sample['global_keep_unify_ratio'])
        return sample

class RandomCrop_Local_spherical:
    def __init__(self):
        pass

    @staticmethod
    def crop(points, p_keep):
        random_point = points[np.random.choice(points.shape[0], 1, replace=False),:3]
        ckt = spt.cKDTree(points)
        knn_dist, knn_idx = ckt.query(random_point, int(np.ceil(points.shape[0]*p_keep)))
        crop_point = points[knn_idx[0]]
        return crop_point

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = self.crop(sample['points'], sample['local_keep_diff_ratio'] if 'local_keep_diff_ratio' in sample else sample['local_keep_unify_ratio'])
        return sample


class RandomCrop_Local_dense:
    """Randomly crops the *source* point cloud, approximately retaining half the points

    A direction is randomly sampled from S2, and we retain points which lie within the
    half-space oriented in this direction.
    If p_keep != 0.5, we shift the plane until approximately p_keep points are retained
    """
    def __init__(self):
        pass

    @staticmethod
    def crop(points, p_keep):
        rand_xyz = uniform_2_sphere()
        centroid = np.mean(points[:, :3], axis=0)
        points_centered = points[:, :3] - centroid

        dist_from_plane = np.dot(points_centered, rand_xyz)
        if p_keep == 0.5:
            mask = dist_from_plane > 0
        else:
            mask = dist_from_plane > np.percentile(dist_from_plane, (1.0 - p_keep) * 100)
            # 消除裁减后点数不是一样的问题，主要在SHAPENET中存在
            if sum(mask)<int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==False)[0].tolist()[-(int(np.ceil(p_keep*points.shape[0]))-sum(mask)):]]=True
            elif sum(mask)>int(np.ceil(p_keep*points.shape[0])):
                mask[np.where(mask==True)[0].tolist()[-(sum(mask)-int(np.ceil(p_keep*points.shape[0]))):]]=False

        return points[mask, :]

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = self.crop(sample['points'], sample['local_keep_diff_ratio'] if 'local_keep_diff_ratio' in sample else sample['local_keep_unify_ratio'])
        return sample


class RandomCrop_sparse(object):
    """
    Random_point_dropout.
    """
    def __init__(self, dropout_ratio):
        self.min_dropout_ratio = dropout_ratio[0]
        self.max_dropout_ratio = dropout_ratio[1]

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        dropout_ratio = np.random.uniform(self.min_dropout_ratio, self.max_dropout_ratio)  # 0~0.875
        drop_idx = np.random.choice(np.arange(sample['points'].shape[0]), np.floor(dropout_ratio*sample['points'].shape[0]).int())
        # drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        # print ('use random drop', len(drop_idx))

        if len(drop_idx) > 0:
            sample['points'][drop_idx, :] = sample['points'][0, :]  # set to the first point
        return sample


class Random_global_keep_ratio(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['global_keep_diff_ratio'] = np.random.uniform(sample['global_keep'][0], sample['global_keep'][1])
        return sample

class Random_local_keep_ratio(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['local_keep_diff_ratio'] = np.random.uniform(sample['local_keep'][0], sample['local_keep'][1])
        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""
    def __call__(self, sample):
        sample['deterministic'] = True
        return sample


class ShufflePoints:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        sample['points'] = np.random.permutation(sample['points'])
        return sample

class OutputPoints:
    """Swap the number of feature channels with the number of points and output points"""
    def __call__(self, sample):
        return sample['points'].transpose(1,0)

class OutputRawPoints:
    """Swap the number of feature channels with the number of points and output raw points"""
    def __call__(self, sample):
        return sample['points_raw'].transpose(1,0)

class CompletionGlobalPoints_repeat:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        if sample['points'].shape[0] < sample['global_keep_len']:
            rand_idxs = np.concatenate([np.random.choice(sample['points'].shape[0],
                                                         sample['points'].shape[0], replace=False),
                                        np.random.choice(sample['points'].shape[0],
                                                         sample['global_keep_len'] - sample['points'].shape[0], replace=True)])
        else:
            rand_idxs = np.random.choice(sample['points'].shape[0], sample['global_keep_len'], replace=False)
        sample['points'] = sample['points'][rand_idxs, :]
        return sample

class CompletionGlobalPoints_interp:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        if sample['points'].shape[0] < sample['global_keep_len']:
            while sample['points'].shape[0] < sample['global_keep_len']:
                ckt = spt.cKDTree(sample['points'])
                knn_dist, knn_idx = ckt.query(sample['points'], 4)
                interp_points = sample['points'][knn_idx].mean(1)
                diff_num = sample['global_keep_len'] - sample['points'].shape[0]
                if diff_num>sample['points'].shape[0]:
                    diff_num = sample['points'].shape[0]
                interp_points_rand_idxs = np.random.choice(sample['points'].shape[0], diff_num, replace=False)
                sample['points'] = np.concatenate([sample['points'], interp_points[interp_points_rand_idxs]])
        else:
            rand_idxs = np.random.choice(sample['points'].shape[0], sample['global_keep_len'], replace=False)
            sample['points'] = sample['points'][rand_idxs, :]
        return sample

class CompletionlocalPoints_repeat:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        if sample['points'].shape[0] < sample['local_keep_len']:
            rand_idxs = np.concatenate([np.random.choice(sample['points'].shape[0],
                                                         sample['points'].shape[0], replace=False),
                                        np.random.choice(sample['points'].shape[0],
                                                         sample['local_keep_len'] - sample['points'].shape[0], replace=True)])
        else:
            rand_idxs = np.random.choice(sample['points'].shape[0], sample['local_keep_len'], replace=False)
        sample['points'] = sample['points'][rand_idxs, :]
        return sample

class CompletionlocalPoints_interp:
    """Shuffles the order of the points"""
    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        if sample['points'].shape[0] < sample['local_keep_len']:
            while sample['points'].shape[0] < sample['local_keep_len']:
                ckt = spt.cKDTree(sample['points'])
                knn_dist, knn_idx = ckt.query(sample['points'], 4)
                interp_points = sample['points'][knn_idx].mean(1)
                diff_num = sample['local_keep_len'] - sample['points'].shape[0]
                if diff_num>sample['points'].shape[0]:
                    diff_num = sample['points'].shape[0]
                interp_points_rand_idxs = np.random.choice(sample['points'].shape[0], diff_num, replace=False)
                sample['points'] = np.concatenate([sample['points'], interp_points[interp_points_rand_idxs]])
        else:
            rand_idxs = np.random.choice(sample['points'].shape[0], sample['local_keep_len'], replace=False)
            sample['points'] = sample['points'][rand_idxs, :]
        return sample


class RandomDiffLevelFps(object):
    """
    We use FPs to sample point clouds at different levels
    """
    def __init__(self, fps_num=128):
        self.fps_num=fps_num

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [B, N, 3]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [B, npoint]
        """
        N, C = xyz.shape
        centroids = np.zeros(npoint).astype(int)
        distance = np.ones(N) * 1e10
        farthest = np.random.randint(0, N, 1)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :]
            dist = np.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = np.argmax(distance)
        return centroids

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        fps_ids = self.farthest_point_sample(sample['points'], self.fps_num)
        sample['points'] = sample['points'][fps_ids]
        return sample


class TranslateAndScalePoints(object):
    """
    Translate and sacle pointcloud.
    """
    def __init__(self, scale_low=2./3., scale_high=3./2., t_range=0.2):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.t_range = t_range

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        scale = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
        translation = np.random.uniform(low=-1*self.t_range, high=self.t_range, size=[3])

        sample['points'] = np.add(np.multiply(sample['points'], scale), translation).astype('float32')
        return sample


class TranslatePoints(object):
    """
    Translate pointcloud.
    """
    def __init__(self, t_range=0.2):
        self.t_range = t_range

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        translation = np.random.uniform(low=-1*self.t_range, high=self.t_range, size=[3])

        sample['points'] = np.add(sample['points'], translation).astype('float32')
        return sample


class ScalePoints(object):
    """
    Scale pointcloud.
    """
    def __init__(self, scale_low=2./3., scale_high=3./2.):
        self.scale_low = scale_low
        self.scale_high = scale_high

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        scale = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])

        sample['points'] = np.multiply(sample['points'], scale).astype('float32')
        return sample


class Jitter_Points(object):
    """
    Jitter_Points.
    """
    def __init__(self, sigma=0.01, clip=0.02):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        N, C = sample['points'].shape
        sample['points'] += np.clip(self.sigma * np.random.randn(N, C), -1 * self.clip, self.clip)
        return sample['points']


class Shuffle(object):
    """
    Shuffle.
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        if 'deterministic' in sample:
            np.random.seed(sample['idx'])
        return np.random.shuffle(sample['points'])


def resample(points, sample_num, deterministic=False, idx=0):
    """Resamples the points such that there is exactly k points.

    If the input point cloud has <= k points, it is guaranteed the
    resampled point cloud contains every point in the input.
    If the input point cloud has > k points, it is guaranteed the
    resampled point cloud does not contain repeated point.
    """

    if deterministic:
        np.random.seed(idx)
    if sample_num <= points.shape[0]:
        rand_idxs = np.random.choice(points.shape[0], sample_num, replace=False)
        return points[rand_idxs, :]
    elif points.shape[0] == sample_num:
        return points
    else:
        rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                    np.random.choice(points.shape[0], sample_num - points.shape[0], replace=True)])
        return points[rand_idxs, :]

def resample_firstn(points, sample_num, deterministic=False, idx=0):
    points = points[:sample_num]
    return points

def random_crop_ratio(p_keep, deterministic=False, idx=0):
    if deterministic:
        np.random.seed(idx)
    keep_ratio = np.random.uniform(p_keep[0], p_keep[1])
    return keep_ratio


class DataAugmentationDINO(object):
    def __init__(self, local_crops_number, diff_resolution=None, diff_resolution_num=None, global2_weakly_aug=False):
        # raw point
        self.raw_transfo1 = transforms.Compose([
            SaveRawPoint(),
            OutputRawPoints()
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            SaveRawPoint(),
            RandomJitter(),
            Random_global_keep_ratio(),
            RandomCrop_Global_spherical(),
            ScalePoints(),
            TranslatePoints(),
            CompletionGlobalPoints_interp(),
            ShufflePoints(),
            OutputPoints()
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            SaveRawPoint(),
            RandomJitter(0.002,0.01) if global2_weakly_aug else RandomJitter(),
            Random_global_keep_ratio(),
            RandomCrop_Global_spherical(),
            ScalePoints(4./5., 5./4.) if global2_weakly_aug else ScalePoints(),
            TranslatePoints(0.05)if global2_weakly_aug else TranslatePoints(),
            CompletionGlobalPoints_interp(),
            ShufflePoints(),
            OutputPoints()
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            SaveRawPoint(),
            RandomJitter(),
            Random_local_keep_ratio(),
            RandomCrop_Local_spherical(),
            ScalePoints(),
            TranslatePoints(),
            CompletionlocalPoints_interp(),
            ShufflePoints(),
            OutputPoints()
        ])
        # transformation for the diff resolution point cloud
        self.use_diff_level_pc = False
        if diff_resolution is not None and diff_resolution_num is not None:
            self.use_diff_level_pc = True
            self.diff_resolution_num = sum(diff_resolution_num)
            self.diff_resolution_transfo = []
            for resolution_i, resolution_i_count in zip(diff_resolution, diff_resolution_num):
                for resolution_i_idx in range(resolution_i_count):
                    self.diff_resolution_transfo_i = transforms.Compose([
                        SaveRawPoint(),
                        RandomJitter(),
                        RandomDiffLevelFps(resolution_i),
                        RandomTranslation(),
                        ShufflePoints(),
                        OutputPoints()
                    ])
                    self.diff_resolution_transfo.append(self.diff_resolution_transfo_i)

    def __call__(self, pc):
        crops = []
        crops.append(self.raw_transfo1(pc))
        crops.append(self.global_transfo1(pc))
        crops.append(self.global_transfo2(pc))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(pc))
        if self.use_diff_level_pc:
            for _ in range(self.diff_resolution_num):
                crops.append(self.diff_resolution_transfo[_](pc))
        return crops

class DataAugmentationSupervisedTrain(object):
    def __init__(self):
        self.raw_transfo1 = transforms.Compose([
            SaveRawPoint(),
            ScalePoints(),
            TranslatePoints(),
            ShufflePoints(),
            OutputPoints()
        ])
    def __call__(self, pc):
        crops = []
        crops.append(self.raw_transfo1(pc))
        return crops

class DataAugmentationSupervisedTest(object):
    def __init__(self):
        # raw point
        self.raw_transfo1 = transforms.Compose([
            SetDeterministic(),
            SaveRawPoint(),
            OutputPoints()
        ])
    def __call__(self, pc):
        crops = []
        crops.append(self.raw_transfo1(pc))
        return crops

## SS = Self-Supervised
class DataAugmentationSSTrain(DataAugmentationSupervisedTrain):
    def __call__(self, pc):
        return self.raw_transfo1(pc)

class DataAugmentationSSTest(DataAugmentationSupervisedTest):
    def __call__(self, pc):
        return self.raw_transfo1(pc)


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', transform=None, deterministic=False, global_keep=None, local_keep=None, use_firstn=False):
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        self.deterministic = deterministic
        if global_keep==None:
            self.global_keep = np.array([0.8, 1.0])
        else:
            self.global_keep = global_keep
        if local_keep==None:
            self.local_keep = np.array([0.5, 0.8])
        else:
            self.local_keep = local_keep
        self.use_firstn = use_firstn

    def __getitem__(self, item):
        if self.use_firstn:
            pointcloud = resample_firstn(self.data[item], self.num_points, self.deterministic, item)
        else:
            pointcloud = resample(self.data[item], self.num_points, self.deterministic, item)
        pointcloud_len = pointcloud.shape[0]
        global_keep_unify_ratio = random_crop_ratio(self.global_keep, self.deterministic, item)
        local_keep_unify_ratio = random_crop_ratio(self.local_keep, self.deterministic, item)
        global_keep_len = int(np.ceil(pointcloud_len * self.global_keep[1]))
        local_keep_len = int(np.ceil(pointcloud_len * self.local_keep[1]))
        # global_keep_ratio = 0.9
        # local_keep_ratio = 0.5
        label = self.label[item]
        sample = {'points_raw': pointcloud, 'label': label, 'idx':np.array(item, dtype=np.int32),
                  'global_keep': self.global_keep, 'local_keep': self.local_keep,
                  'global_keep_unify_ratio': global_keep_unify_ratio, 'local_keep_unify_ratio': local_keep_unify_ratio,
                  'global_keep_len':global_keep_len, 'local_keep_len': local_keep_len}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return self.data.shape[0]


class ModelNet40_STRL(Dataset):
    def __init__(self, num_points, partition='train', transform=None, deterministic=False, global_keep=None, local_keep=None, use_firstn=False):
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        self.deterministic = deterministic
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'data')
        self._cache = os.path.join(DATA_DIR, "modelnet40_normal_resampled_cache")
        if global_keep==None:
            self.global_keep = np.array([0.8, 1.0])
        else:
            self.global_keep = global_keep
        if local_keep==None:
            self.local_keep = np.array([0.5, 0.8])
        else:
            self.local_keep = local_keep
        self.use_firstn = use_firstn

        self._lmdb_file = os.path.join(self._cache, "train" if self.partition=='train' else "test")
        with lmdb.open(self._lmdb_file, map_size=1 << 36) as lmdb_env:
            self._len = lmdb_env.stat()["entries"]

        self._lmdb_env = lmdb.open(self._lmdb_file, map_size=1 << 36, readonly=True, lock=False)

        datalist = []
        labellist = []
        for eleid in range(self._len):
            with self._lmdb_env.begin(buffers=True) as txn:
                ele = msgpack_numpy.unpackb(txn.get(str(eleid).encode()), raw=False)
                datalist.append(ele['pc'][None,:,:])
                labellist.append(np.array([ele['lbl']])[None,:])
        self.data = np.concatenate(datalist)
        self.label = np.concatenate(labellist)

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        point_set, label = self.data[item][:, :3], self.label[item]

        if self.use_firstn:
            pointcloud = resample_firstn(point_set, self.num_points, self.deterministic, item)
        else:
            pointcloud = resample(point_set, self.num_points, self.deterministic, item)
        pointcloud = self.pc_normalize(pointcloud)
        pointcloud_len = pointcloud.shape[0]
        global_keep_unify_ratio = random_crop_ratio(self.global_keep, self.deterministic, item)
        local_keep_unify_ratio = random_crop_ratio(self.local_keep, self.deterministic, item)
        global_keep_len = int(np.ceil(pointcloud_len * self.global_keep[1]))
        local_keep_len = int(np.ceil(pointcloud_len * self.local_keep[1]))
        # global_keep_ratio = 0.9
        # local_keep_ratio = 0.5
        sample = {'points_raw': pointcloud, 'label': label, 'idx':np.array(item, dtype=np.int32),
                  'global_keep': self.global_keep, 'local_keep': self.local_keep,
                  'global_keep_unify_ratio': global_keep_unify_ratio, 'local_keep_unify_ratio': local_keep_unify_ratio,
                  'global_keep_len':global_keep_len, 'local_keep_len': local_keep_len}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return self._len


class ShapeNet(Dataset):
    def __init__(self, num_points, partition='train', transform=None, deterministic=False, global_keep=None, local_keep=None, use_firstn=False):
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.train_npy = os.path.join(BASE_DIR, "data/shapenet57448xyzonly.npz")
        self.td = dict(np.load(self.train_npy))
        # self.train_npy = os.path.join(BASE_DIR, "data/test.npy")
        # self.td = np.load(self.train_npy,allow_pickle=True).item()
        self.data = self.td["data"][:,:,:3]
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        self.deterministic = deterministic
        if global_keep==None:
            self.global_keep = np.array([0.8, 1.0])
        else:
            self.global_keep = global_keep
        if local_keep==None:
            self.local_keep = np.array([0.5, 0.8])
        else:
            self.local_keep = local_keep
        self.use_firstn = use_firstn

    def pc_normalize(self, pc):
        l = pc.shape[0]
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
        pc = pc / m
        return pc

    def __getitem__(self, item):
        if self.use_firstn:
            pointcloud = resample_firstn(self.data[item], self.num_points, self.deterministic, item)
        else:
            pointcloud = resample(self.data[item], self.num_points, self.deterministic, item)
        # pointcloud = self.pc_normalize(pointcloud) 不加更好
        pointcloud_len = pointcloud.shape[0]
        global_keep_unify_ratio = random_crop_ratio(self.global_keep, self.deterministic, item)
        local_keep_unify_ratio = random_crop_ratio(self.local_keep, self.deterministic, item)
        global_keep_len = int(np.ceil(pointcloud_len * self.global_keep[1]))
        local_keep_len = int(np.ceil(pointcloud_len * self.local_keep[1]))
        # global_keep_ratio = 0.9
        # local_keep_ratio = 0.5
        sample = {'points_raw': pointcloud, 'idx':np.array(item, dtype=np.int32),
                  'global_keep': self.global_keep, 'local_keep': self.local_keep,
                  'global_keep_unify_ratio': global_keep_unify_ratio, 'local_keep_unify_ratio': local_keep_unify_ratio,
                  'global_keep_len':global_keep_len, 'local_keep_len': local_keep_len}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0

    def __len__(self):
        return self.data.shape[0]


def load_data_semseg(partition, test_area, train_area):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if partition == 'train':
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data')
    else:
        data_dir = os.path.join(DATA_DIR, 'indoor3d_sem_seg_hdf5_data_test')
    with open(os.path.join(data_dir, "all_files.txt")) as f:
        all_files = [line.rstrip() for line in f]
    with open(os.path.join(data_dir, "room_filelist.txt")) as f:
        room_filelist = [line.rstrip() for line in f]
    data_batchlist, label_batchlist = [], []
    for f in all_files:
        file = h5py.File(os.path.join(DATA_DIR, f), 'r+')
        data = file["data"][:]
        label = file["label"][:]
        data_batchlist.append(data)
        label_batchlist.append(label)
    data_batches = np.concatenate(data_batchlist, 0)
    seg_batches = np.concatenate(label_batchlist, 0)
    test_area_name = "Area_" + test_area
    train_idxs, test_idxs = [], []
    for i, room_name in enumerate(room_filelist):
        if test_area_name in room_name:
            test_idxs.append(i)
        else:
            for area in train_area:
                if "Area_" + area in room_name:
                    train_idxs.append(i)
                    break
    if partition == 'train':
        all_data = data_batches[train_idxs, ...]
        all_seg = seg_batches[train_idxs, ...]
    else:
        all_data = data_batches[test_idxs, ...]
        all_seg = seg_batches[test_idxs, ...]
    return all_data, all_seg


class S3DIS(Dataset):
    def __init__(self, num_points, partition='train', transform=None, deterministic=False, global_keep=None, local_keep=None, use_firstn=False):
        train_area = ['1', '2', '3', '4', '5',]
        test_area = '6'
        self.data, self.seg = load_data_semseg(partition, test_area, train_area)
        self.num_points = num_points
        self.partition = partition
        self.transform = transform
        self.deterministic = deterministic
        if global_keep==None:
            self.global_keep = np.array([0.8, 1.0])
        else:
            self.global_keep = global_keep
        if local_keep==None:
            self.local_keep = np.array([0.5, 0.8])
        else:
            self.local_keep = local_keep
        self.use_firstn = use_firstn

    def __getitem__(self, item):
        if self.use_firstn:
            pointcloud = resample_firstn(self.data[item], self.num_points, self.deterministic, item)
        else:
            pointcloud = resample(self.data[item], self.num_points, self.deterministic, item)
        # pointcloud = self.pc_normalize(pointcloud) 不加更好
        pointcloud_len = pointcloud.shape[0]
        global_keep_unify_ratio = random_crop_ratio(self.global_keep, self.deterministic, item)
        local_keep_unify_ratio = random_crop_ratio(self.local_keep, self.deterministic, item)
        global_keep_len = int(np.ceil(pointcloud_len * self.global_keep[1]))
        local_keep_len = int(np.ceil(pointcloud_len * self.local_keep[1]))
        # global_keep_ratio = 0.9
        # local_keep_ratio = 0.5
        sample = {'points_raw': pointcloud, 'idx':np.array(item, dtype=np.int32),
                  'global_keep': self.global_keep, 'local_keep': self.local_keep,
                  'global_keep_unify_ratio': global_keep_unify_ratio, 'local_keep_unify_ratio': local_keep_unify_ratio,
                  'global_keep_len':global_keep_len, 'local_keep_len': local_keep_len}
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, 0

    def __len__(self):
        return self.data.shape[0]