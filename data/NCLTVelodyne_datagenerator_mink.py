import os
import numpy as np
import os.path as osp
import h5py
import torch
import MinkowskiEngine as ME
import pypatchworkpp
import open3d as o3d
from data.robotcar_sdk.python.velodyne import load_velodyne_binary_seg, get_velo
from torch.utils import data
from utils.pose_util import filter_overflow_nclt, interpolate_pose_nclt, so3_to_euler_nclt, process_poses, cartesian_to_polar_expansion, polar_expansion_to_cartesian

BASE_DIR = osp.dirname(osp.abspath(__file__))


class NCLT_mink(data.Dataset):
    def __init__(
        self,
        data_path,
        train=True,
        voxel_size=0.3,
        min_range=1.0,
        max_range=100.0,
        horizontal_res=1024,
        level_correction=False
    ):
        # directories
        data_dir = osp.join(data_path, 'NCLT')
        self.voxel_size = voxel_size
        self.min_range = min_range
        self.max_range = max_range
        self.horizontal_res = horizontal_res
        self.level_correction = level_correction

        # decide which sequences to use
        if train:
            seqs = ["2012-01-22", "2012-02-02", "2012-02-18", "2012-05-11"]
        else:
            seqs = ["2012-02-12", "2012-02-19", "2012-03-31", "2012-05-26"]
            # seqs = ["2012-02-12"]
            # seqs = ["2012-02-19"]
            # seqs = ["2012-03-31"]
            # seqs = ["2012-05-26"]

        ps = {}
        ts = {}
        vo_stats = {}
        self.pcs = []
        for seq in seqs:
            seq_dir = osp.join(data_dir, seq )
            # read the image timestamps
            print('interpolate ' + seq)
            ts_raw = []
            # 读入LiDAR时间戳，并从小到大排序
            
            vel = os.listdir(seq_dir + '/velodyne_sync')
            
            for i in range(len(vel)):
                ts_raw.append(int(vel[i][:-4]))
            ts_raw = sorted(ts_raw)
            # GT poses
            gt_filename = osp.join(seq_dir, 'groundtruth_'+ seq + '.csv')
            ts[seq] = filter_overflow_nclt(gt_filename, ts_raw)
            p = interpolate_pose_nclt(gt_filename, ts[seq])  # (n, 6)
            p = so3_to_euler_nclt(p)   # (n, 4, 4)
            ps[seq] = np.reshape(p[:, :3, :], (len(p), -1))  # (n, 12)

            vo_stats[seq] = {'R': np.eye(3), 't': np.zeros(3), 's': 1}

            self.pcs.extend([osp.join(seq_dir, 'velodyne_sync', '{:d}.bin'.format(t)) for t in ts[seq]])

        # convert the pose to translation + log quaternion, align, normalize
        self.poses = np.empty((0, 6))
        self.rots = np.empty((0, 3, 3))
        for seq in seqs:
            pss, rotation, pss_max, pss_min = process_poses(poses_in=ps[seq], mean_t=0., std_t=0.,
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
        params.verbose = True
        self.patchworkpp = pypatchworkpp.patchworkpp(params)

    def get_center_t(self):
        return self.center_t

    def __getitem__(self, index):
        scan_path = self.pcs[index]
        transition = self.poses[index, :3]  # (6,)
        rot = self.rots[index]
        scan, label = get_velo(scan_path)
        range3d = np.linalg.norm(scan, axis=1)
        mask = (range3d > self.min_range) & (range3d < self.max_range)
        scan = scan[mask]
        label = label[mask]

        transform = np.eye(4)
        transform[:3, :3] = rot
        transform[:3, 3] = transition

        label = label[..., np.newaxis]

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
                num_iterations = 100,
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
        # ground truth
        pl_feats = np.concatenate((highs, ranges, label), axis=1)
        coords, feats = ME.utils.sparse_quantize(
            coordinates=pl_coords,
            features=pl_feats,
            quantization_size=self.voxel_size,
        )

        return coords, feats, scan, transform, correction

    def __len__(self):
        return len(self.poses)
