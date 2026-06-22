import os
import sys
import torch
import numpy as np
import pickle
import os.path as osp
import h5py
import json
import MinkowskiEngine as ME
import pypatchworkpp
import open3d as o3d
from data.robotcar_sdk.python.interpolate_poses import interpolate_ins_poses, interpolate_vo_poses
from data.robotcar_sdk.python.transform import build_se3_transform
from data.robotcar_sdk.python.velodyne import load_velodyne_binary
from torch.utils import data
from utils.pose_util import calibrate_process_poses, filter_overflow_ts, cartesian_to_polar_expansion
from copy import deepcopy

BASE_DIR = osp.dirname(osp.abspath(__file__))


class RobotCar_mink(data.Dataset):
    def __init__(self,
        data_path,
        train=True,
        voxel_size=0.3,
        min_range=1.0,
        max_range=100.0,
        horizontal_res=1024,
        level_correction=True,
    ):
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range
        self.horizontal_res = horizontal_res
        self.level_correction = level_correction

        # directories
        lidar = 'velodyne_left'
        data_dir = osp.join(data_path, 'Oxford')

        # decide which sequences to use
        if train:
            seqs = ['2019-01-11-14-02-26', '2019-01-14-12-05-52', '2019-01-14-14-48-55', '2019-01-18-15-20-12']
        else:
            seqs = ['2019-01-15-13-06-37', '2019-01-17-13-26-39', '2019-01-17-14-03-00', '2019-01-18-14-14-42']
            # seqs = ['2019-01-15-13-06-37']
            # seqs = ['2019-01-17-13-26-39']
            # seqs = ['2019-01-17-14-03-00']
            # seqs = ['2019-01-18-14-14-42']

        ps = {}
        ts = {}
        vo_stats = {}
        pcs_all = []
        self.pcs = []

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq + '-radar-oxford-10k')
            print('interpolate ' + seq)
            ts_filename = osp.join(seq_dir, lidar + '.timestamps')
            with open(ts_filename, 'r') as f:
                ts_raw = [int(l.rstrip().split(' ')[0]) for l in f]
            ins_filename = osp.join(seq_dir, 'gps', 'ins.csv')
            ts[seq] = filter_overflow_ts(ins_filename, ts_raw)
            rot = np.fromfile(osp.join(seq_dir, 'rot_tr.bin'), dtype=np.float32).reshape(-1, 9)
            t   = np.fromfile(osp.join(seq_dir, 'tr_add_mean.bin'), dtype=np.float32).reshape(-1, 3)
            ps[seq] = np.concatenate((rot, t[:, :3]), axis=1) # (n, 12)
            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.pcs.extend([osp.join(seq_dir, 'velodyne_left', '{:d}.bin'.format(t)) for t in ts[seq]])

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))

        for seq in seqs:
            pss, rotation, pss_max, pss_min = calibrate_process_poses(poses_in=ps[seq], mean_t=0.,
                                                            align_R=vo_stats[seq]['R'], align_t=vo_stats[seq]['t'],
                                                            align_s=vo_stats[seq]['s'])
            self.poses = np.vstack((self.poses, pss))
            self.rots = np.vstack((self.rots, rotation))

        self.center_t = np.concatenate([np.mean(self.poses[:, :2], axis=0), np.min(self.poses[:, 2:3], axis=0)])

        if train:
            print("train data num:" + str(len(self.poses)))
        else:
            print("valid data num:" + str(len(self.poses)))

        # Patchwork++ initialization
        params = pypatchworkpp.Parameters()
        # params.verbose = True
        self.patchworkpp = pypatchworkpp.patchworkpp(params)


    def get_center_t(self):
        return self.center_t

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        ptcld = load_velodyne_binary(scan_path)  # (N, 4)
        ptcld = ptcld.transpose()
        ptcld[:, 2] *= -1
        range3d = np.linalg.norm(ptcld[:, :3], axis=1)
        mask = (range3d > self.min_range) & (range3d < self.max_range)
        scan = ptcld[mask, :3]  # (N, 3)
        label = ptcld[mask, 3:4]

        pose = self.poses[index]  # (6,)
        rot = self.rots[index]
        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = pose[:3]

        correction = np.eye(4)
        if self.level_correction:
            # segment ground
            self.patchworkpp.estimateGround(np.concatenate([scan, label], axis=-1))
            ground_idx = self.patchworkpp.getGroundIndices()
            nonground_idx = self.patchworkpp.getNongroundIndices()
            ground = scan[ground_idx]
            nonground = scan[nonground_idx]
            pcd_ground = o3d.geometry.PointCloud()
            pcd_ground.points = o3d.utility.Vector3dVector(ground)
            # fit plane
            plane_model, inliers = pcd_ground.segment_plane(
                distance_threshold = 0.1,
                ransac_n = 3,
                num_iterations = 1000,
            )
            # ajust pointcloud
            a, b, c, d = plane_model
            normal = np.array([a, b, c])
            z_axis = np.array([0, 0, 1])
            rotation_axis = np.cross(normal, z_axis)
            rotation_axis /= np.linalg.norm(rotation_axis)
            angle = np.arccos(np.dot(normal, z_axis) / (np.linalg.norm(normal) * np.linalg.norm(z_axis)))
            rot_plane = o3d.geometry.get_rotation_matrix_from_axis_angle(rotation_axis * angle)
            distance = d / np.linalg.norm(normal)
            t_plane = np.array([0, 0, distance], dtype=np.float32)
            trans_plane = np.eye(4)
            trans_plane[:3, :3] = rot_plane
            trans_plane[:3, 3] = t_plane
            pcd_ori = o3d.geometry.PointCloud()
            pcd_ori.points = o3d.utility.Vector3dVector(scan[nonground_idx, :3])
            pcd_ori.paint_uniform_color([1, 0, 0])
            pcd_orig = o3d.geometry.PointCloud()
            pcd_orig.points = o3d.utility.Vector3dVector(scan[ground_idx, :3])
            pcd_orig.paint_uniform_color([1, 0.6, 0.6])

            correction = trans_plane
            scan[:, :3] = (rot_plane @ scan[:, :3].T + t_plane[:, np.newaxis]).T
            
        pl_coords = cartesian_to_polar_expansion(scan, self.voxel_size*self.horizontal_res)
        ranges = pl_coords[:, 1:2]
        highs = pl_coords[:, 2:3]
        pl_feats = np.concatenate((highs, ranges, label), axis=1)

        coords, feats = ME.utils.sparse_quantize(
            coordinates=pl_coords,
            features=pl_feats,
            quantization_size=self.voxel_size)

        return coords, feats, scan, transform, correction

    def __len__(self):
        return len(self.poses)