# pylint: disable=no-member
import argparse
from contextlib import nullcontext
import json
import os
import shutil
import sys
import threading
import traceback
from typing import Callable
import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
import open3d as o3d
from tqdm import tqdm
from torch.profiler import profile, record_function, ProfilerActivity
from utils.pose_util import estimate_poses
import transforms3d.quaternions as txq

from torch.cuda.amp import autocast
from models.sc2pcr import Matcher

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data.Enhanced_OxfordVelodyne_datagenerator_mink import  RobotCar_mink as RobotCar
from data.NCLTVelodyne_datagenerator_mink import NCLT_mink as NCLT
from models.model_mink import LEADER
from utils.pose_util import val_translation, val_rotation, qexp, estimate_poses, polar_expansion_to_cartesian
from tensorboardX import SummaryWriter
from torch import Tensor
from torch.backends import cudnn
from torch.utils.data import DataLoader
import torch.nn.utils.rnn as rnn_utils
from os import path as osp
import time
from accelerate import Accelerator
import MinkowskiEngine as ME

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
cudnn.enabled = True

def get_args(is_main_process=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=25,
                        help='Oxford 35 NCLT 30')
    parser.add_argument('--val_batch_size', type=int, default=50,
                        help='Batch Size during validating [default: 80]')
    parser.add_argument('--max_epoch', type=int, default=50,
                        help='Epoch to run [default: 100]')
    parser.add_argument('--max_encoder_epoch', type=int, default=100,
                        help='Epoch to run encoder [default: 30]')
    parser.add_argument('--init_learning_rate', type=float, default=0.001,
                        help='Initial learning rate [default: 0.001]')
    parser.add_argument("--decay_epoch", type=float, default=1,
                        help="Oxford: 1200 NCLT: 1000")
    parser.add_argument('--seed', type=int, default=20, metavar='S',
                        help='random seed (default: 20)')
    parser.add_argument('--mode', type=str, default="test",
                        help='train or test [default: train]')
    parser.add_argument('--log_dir', default='',
                        help='Log dir [default: log]')
    parser.add_argument('--dataset_folder', default='',
                        help='Our Dataset Folder') # /ava16t/lw/Data
    parser.add_argument('--dataset', default='NCLT',
                        help='Oxford or NCLT')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num workers for dataloader, default:2')
    parser.add_argument('--horizontal_res', type=float, default=1024,
                        help='Oxford 1024 NCLT: 1024')
    parser.add_argument('--voxel_size', type=float, default=0.2,
                        help='voxel size in direction of y, z. default: Oxford 0.2, NCLT 0.2')
    parser.add_argument('--max_range', type=float, default=100,
                        help='max range of points, default: Oxford 100, NCLT 100')
    parser.add_argument('--resume_model', type=str, default='',
                        help='If present, restore checkpoint and resume training')
    parser.add_argument('--profile', action='store_true', default=False,
                        help='Enable torch profiler and export chrome trace')
    parser.add_argument('--profile_out', type=str, default='',
                        help='Chrome trace output path (default: log_dir/trace_epoch{N}.json)')

    FLAGS = parser.parse_args()
    args = vars(FLAGS)
    if not FLAGS.log_dir:
        FLAGS.log_dir = 'log_' + FLAGS.dataset + '/'

    return FLAGS


def get_data_loader(FLAGS):
    if FLAGS.dataset == 'Oxford':
        dataset_class = RobotCar
    elif FLAGS.dataset == 'NCLT':
        dataset_class = NCLT
    else:
        raise ValueError('Dataset not found!')

    train_set = dataset_class(
        data_path=FLAGS.dataset_folder,
        train=True,
        voxel_size=FLAGS.voxel_size,
        horizontal_res=FLAGS.horizontal_res
    )

    val_set = dataset_class(
        data_path=FLAGS.dataset_folder,
        train=False,
        voxel_size=FLAGS.voxel_size,
        horizontal_res=FLAGS.horizontal_res,
        # reverse=True
    )

    def collate_pair_fn(list_data):
        N = len(list_data)
        list_data = [data for data in list_data if data is not None]

        coords, feats, points, T, T_corr = list(zip(*list_data))

        coords_batch = ME.utils.batched_coordinates(coords)
        feats_batch = torch.from_numpy(np.concatenate(feats)).float()
        T_batch = torch.from_numpy(np.stack(T)).float()
        T_corr_batch = torch.from_numpy(np.stack(T_corr)).float()

        return {
            'coords': coords_batch,
            'feats': feats_batch,
            'points': points,
            'T': T_batch,
            'T_corr': T_corr_batch
        }
    collation_fn = collate_pair_fn

    train_loader = DataLoader(train_set,
                            batch_size=FLAGS.batch_size,
                            shuffle=True,
                            collate_fn=collation_fn,
                            num_workers=FLAGS.num_workers,
                            pin_memory=True)

    val_loader = DataLoader(val_set,
                            batch_size=FLAGS.val_batch_size,
                            shuffle=False,
                            collate_fn=collation_fn,
                            num_workers=FLAGS.num_workers,
                            pin_memory=True)
    
    return train_loader, val_loader


class Logger:
    def __init__(self, FLAGS):
        self.LOG_FOUT = open(os.path.join(FLAGS.log_dir, 'log.txt'), 'w')
        self.__call__(str(FLAGS))

    def __call__(self, out_str):
        self.LOG_FOUT.write(out_str + '\n')
        self.LOG_FOUT.flush()
        print(out_str)


class TRR:
    def __init__(self, scale=10.0):
        super().__init__()
        self.scaler = np.log(scale) / np.pi

    def __call__(
        self, 
        gt: Tensor, 
        pred: Tensor,
        u: Tensor, 
        batch_idx: Tensor
    ):
        # prediction loss
        batch_size = batch_idx.max() + 1
        l_raw = (pred - gt).norm(dim=-1)
        u_cut = u.clamp(min=-10*torch.pi, max=10*torch.pi).atan() * self.scaler
        u_scale = u.atan() * self.scaler

        u_cut_exp = u_cut.exp()
        u_scale_exp = u_scale.exp()
        w_sum = torch.zeros((batch_size), dtype=pred.dtype, device=pred.device)\
                .scatter_add_(0, batch_idx, torch.min(u_cut_exp, u_scale_exp))
        w = torch.max(u_cut_exp, u_scale_exp) / \
            w_sum.clamp(min=1e-6)[batch_idx]
        L_TRR = torch.zeros((batch_size), dtype=pred.dtype, device=pred.device)
        L_TRR.scatter_add_(0, batch_idx, l_raw * w).mean()
        L2 = l_raw.mean()
        return L_TRR, L2


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def load_checkpoint(path, accelerator, process_info):
    print("Resuming From ", path)
    accelerator.load_state(path)
    with open(os.path.join(path, 'extra.json'), 'r') as f:
        info = json.load(f)
    process_info['epoch'] = info['epoch']
    process_info['train_iter'] = info['train_iter']
    process_info['val_iter'] = info['val_iter']
    process_info['center_t'] = info['center_t']


def save_checkpoint(path, accelerator, process_info):
    accelerator.save_state(path)
    with open(os.path.join(path, 'extra.json'), 'w') as f:
        json.dump(
            {
                'epoch': process_info['epoch'],
                'train_iter': process_info['train_iter'],
                'val_iter': process_info['val_iter'],
                'center_t': process_info['center_t']
            },
            f
        )
    print("save checkpoint_epoch_{}".format(process_info['epoch']))


def train():
    FLAGS = get_args()
    accelerator = Accelerator()
    if accelerator.state.is_main_process:
        os.makedirs(FLAGS.log_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    setup_seed(FLAGS.seed)
    train_loader, val_loader = get_data_loader(FLAGS)

    logger = Logger(FLAGS)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        raise ValueError("GPU not found!")

    train_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'train'))
    val_writer = SummaryWriter(os.path.join(FLAGS.log_dir, 'valid'))
    model = LEADER(in_channels=3, 
                   out_channels=4, 
                   feat_channels=512, 
                   width=FLAGS.horizontal_res)
    loss_fn = TRR(scale=10.0)
    ransac = Matcher(inlier_threshold=3.0,
                     d_thre=2,
                     num_iterations=10,
                     ratio=0.15,
                     nms_radius=0.1,
                     max_points=3000,
                     k1=30)

    optimizer = torch.optim.Adam(
        [
            {'params': model.parameters(), 'lr': FLAGS.init_learning_rate}
        ], 
        FLAGS.init_learning_rate
    )

    center_t = train_loader.dataset.get_center_t() + np.array([0, 0, -2 * FLAGS.max_range])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, FLAGS.decay_epoch, gamma=0.9)
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(model, optimizer, train_loader, val_loader, scheduler)
    process_info = {'epoch': -1, 'train_iter': 0, 'val_iter': 0, 'center_t': center_t.tolist()}
    resume_dir = FLAGS.resume_model
    if resume_dir:
        load_checkpoint(resume_dir, accelerator, process_info)
    
    starting_epoch = process_info['epoch'] + 1
    
    start_time = time.time()
    sys.stdout.flush()

    if FLAGS.mode == 'test':
        process_one_epoch(model, val_loader, optimizer, scheduler, accelerator, ransac, process_info, False, val_writer, loss_fn, logger, FLAGS, device)
        exit(0)

    for epoch in range(starting_epoch, FLAGS.max_epoch):
        process_info['epoch'] = epoch
        process_one_epoch(model, train_loader, optimizer, scheduler, accelerator, ransac, process_info, True, train_writer, loss_fn, logger, FLAGS, device)

        if (epoch + 1) % 1 == 0 and accelerator.is_main_process:
            resume_dir = os.path.join(FLAGS.log_dir, 'checkpoint_epoch_{}'.format(epoch))
            save_checkpoint(resume_dir, accelerator, process_info)
            if epoch % 10 != 0 or epoch < 30:
                del_dir = os.path.join(FLAGS.log_dir, 'checkpoint_epoch_{}'.format(epoch - 1))
                if os.path.exists(del_dir):
                    shutil.rmtree(del_dir)

        if (epoch + 1) % 10 == 0:
            process_one_epoch(model, val_loader, optimizer, scheduler, accelerator, ransac, process_info, False, val_writer, loss_fn, logger, FLAGS, device)

    train_writer.close()
    val_writer.close()
    end_time = time.time()
    execution_time = end_time - start_time
    logger(' EPOCH %03d time cost = %03d' % (FLAGS.max_epoch, execution_time))
    sys.stdout.flush()


def process_one_epoch(
    model: nn.Module, 
    data_loader: DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler, 
    accelerator: Accelerator, 
    ransac: any,
    process_info: dict,
    train: bool, 
    summary_writer: SummaryWriter, 
    loss_fn: Callable,
    logger: Logger,
    FLAGS: argparse.Namespace,
    device: torch.device
):
    time_results = []
    loss_sum = 0
    center_t = torch.tensor(process_info['center_t'], dtype=torch.float32, device=device)

    # -- profile: first training epoch, record a few batches --
    if FLAGS.profile and train and process_info['epoch'] == 0:
        prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True)
        prof.start()

    pbar = tqdm(
        total=len(data_loader),
        desc='Epoch %d %s' % (process_info['epoch'], 'train' if train else 'val '),
        disable=not accelerator.is_local_main_process,
        leave=train,
    )

    if train:
        model.train()
        context_manager = nullcontext()
    else:
        model.eval()
        context_manager = torch.no_grad()

    all_pred_T = []
    all_gt_T = []
    for step, input_dict in enumerate(data_loader):

        optimizer.zero_grad()
        
        start = time.time()
        coords = input_dict['coords']
        feats = input_dict['feats']
        points = input_dict['points']
        input = ME.SparseTensor(feats, coords)
        gt_T = input_dict['T']
        T_corr = input_dict['T_corr']

        batch_size = gt_T.shape[0]

        with context_manager:   
            enc = model.encoder(input)
            enc_C = enc.C
            enc_F = enc.F
            stride = torch.tensor(enc.tensor_stride, device=device, dtype=torch.float32)
            voxel_size = FLAGS.voxel_size

            batch_idx = enc_C[:, 0].long()
            gt_T_corr = gt_T @ torch.inverse(T_corr)
            voxel_centers = (enc_C[:, 1:].float() + stride / 2) * voxel_size
            voxel_centers_l = polar_expansion_to_cartesian(voxel_centers, FLAGS.horizontal_res * voxel_size)    # local
            voxel_centers_w = (voxel_centers_l[:, None] @ gt_T_corr[batch_idx, :3, :3].permute(0, 2, 1))[:, 0] + gt_T_corr[batch_idx, :3, 3] - center_t

            pred_f = model.decoder(enc_F)

        if train:
            loss_t_weighted, loss_t = loss_fn(voxel_centers_w,
                                              pred_f[:, :3],
                                              pred_f[:, -1], 
                                              batch_idx)
            loss_sum += loss_t.mean().item()
            loss = loss_t_weighted.mean()

            accelerator.backward(loss)
            optimizer.step()
            end = time.time()
            process_info['train_iter'] += 1
            # profile: collect 5 batches then export
            if FLAGS.profile and train and process_info['epoch'] == 0:
                if step < 5:
                    prof.step()
                elif step == 5:
                    prof.stop()
                    trace_path = FLAGS.profile_out or os.path.join(FLAGS.log_dir, 'trace.json')
                    prof.export_chrome_trace(trace_path)
                    logger(f'Chrome trace saved to: {trace_path}')
            if accelerator.is_local_main_process:
                summary_writer.add_scalar('Loss_t', loss_t.mean().cpu().item(), process_info['train_iter'])
            pbar.set_postfix({'loss': f'{loss_t.mean().item():.4f}'})
            pbar.update(1)
            time_results.append((end - start) / batch_size)

        else:
            pred_T = []
            with autocast(enabled=False):
                for i in range(batch_size):
                    batch_mask = batch_idx == i
                    c_pred = pred_f[batch_mask, :3].float()
                    u_pred = pred_f[batch_mask, 3].float()
                    c_local = voxel_centers_l[batch_mask, :3].float()
                    c_gt = voxel_centers_w[batch_mask, :3].float()

                    # select top-50%
                    _, indices = u_pred.topk(k=max(min(50, u_pred.shape[0]), int(0.5*u_pred.shape[0])))
                    c_pred = c_pred[indices]
                    c_local = c_local[indices]
                    c_gt = c_gt[indices]

                    T = ransac.estimator(c_local[None], c_pred[None])[0]
                    T[:3, 3] += center_t
                    T = T @ (T_corr[i])
                    
                    pred_T.append(T)
            end = time.time()
            process_info['val_iter'] += 1
            pred_T = torch.stack(pred_T, dim=0)  # (local_bs, 4, 4)
            all_pred_T.append(pred_T)
            all_gt_T.append(gt_T)
            time_results.append((end - start) / batch_size)

            # per-batch logging
            pred_T = pred_T.cpu().numpy()
            batch_error_t = np.linalg.norm(pred_T[:, :3, 3] - gt_T[:, :3, 3].cpu().numpy(), axis=1)
            batch_error_q = np.arccos(
                np.clip(
                    (np.trace(
                        np.matmul(pred_T[:, :3, :3].transpose(0, 2, 1), gt_T[:, :3, :3].cpu().numpy()), axis1=1, axis2=2
                    ) - 1) / 2, -1, 1
                )
            ) * 180 / np.pi
            if accelerator.is_local_main_process:
                summary_writer.add_scalar('F_error_R', batch_error_q.mean(), process_info['val_iter'])
                summary_writer.add_scalar('F_error_t', batch_error_t.mean(), process_info['val_iter'])
                frame_start = step * FLAGS.val_batch_size
                frame_end = frame_start + batch_size - 1
                pbar.write(f'Frame {frame_start}-{frame_end} | Mean: {batch_error_t.mean():.4f} m, {batch_error_q.mean():.2f} ° | '
                           f'Median: {np.median(batch_error_t):.4f} m, {np.median(batch_error_q):.2f} °')
            pbar.set_postfix({'t': f'{batch_error_t.mean():.3f}m', 'R': f'{batch_error_q.mean():.1f}°'})
            pbar.update(1)

    pbar.close()

    if train:
        scheduler.step()
        if accelerator.is_local_main_process:
            logger('Epoch %d average loss: %f' % (process_info['epoch'], loss_sum / len(data_loader)))
        return

    # -- validation: compute metrics on collected results --
    if not accelerator.is_main_process:
        return

    all_pred_T = torch.cat(all_pred_T, dim=0).cpu().numpy()
    all_gt_T   = torch.cat(all_gt_T, dim=0).cpu().numpy()

    error_t = np.linalg.norm(all_pred_T[:, :3, 3] - all_gt_T[:, :3, 3], axis=1)
    error_q = np.arccos(
        np.clip(
            (np.trace(
                np.matmul(all_pred_T[:, :3, :3].transpose(0, 2, 1), all_gt_T[:, :3, :3]), axis1=1, axis2=2
            ) - 1) / 2, -1, 1
        )
    ) * 180 / np.pi

    mean_ATE = np.mean(error_t)
    mean_ARE = np.mean(error_q)
    median_ATE = np.median(error_t)
    median_ARE = np.median(error_q)
    mean_time = np.mean(time_results)
    logger('Mean Error: %f m, %f °' % (mean_ATE, mean_ARE))
    logger('Median Error: %f m, %f °' % (median_ATE, median_ARE))
    logger('Mean Cost Time(s): %f' % mean_time)
    summary_writer.add_scalar('MeanATE', mean_ATE, 0)
    summary_writer.add_scalar('MeanARE', mean_ARE, 0)
    summary_writer.add_scalar('MedianATE', median_ATE, 0)
    summary_writer.add_scalar('MedianARE', median_ARE, 0)
    summary_writer.add_scalar('MeanTime', mean_time, 0)

    # trajectory
    fig = plt.figure()
    plt.scatter(all_gt_T[:, 1, 3], all_gt_T[:, 0, 3], s=3, c='black')
    plt.scatter(all_pred_T[:, 1, 3], all_pred_T[:, 0, 3], s=3, c='red')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.plot(all_gt_T[0, 1, 3], all_gt_T[0, 0, 3], 'y*', markersize=10)
    image_filename = os.path.join(os.path.expanduser(FLAGS.log_dir), '{:s}.png'.format('trajectory'))
    fig.savefig(image_filename, dpi=200, bbox_inches='tight')

    # save error and trajectory
    error_t_filename = osp.join(FLAGS.log_dir, 'error_t.txt')
    error_q_filename = osp.join(FLAGS.log_dir, 'error_q.txt')
    pred_t_filename = osp.join(FLAGS.log_dir, 'pred_t.txt')
    gt_t_filename = osp.join(FLAGS.log_dir, 'gt_t.txt')
    np.savetxt(error_t_filename, error_t, fmt='%8.7f')
    np.savetxt(error_q_filename, error_q, fmt='%8.7f')
    np.savetxt(pred_t_filename, all_pred_T[:, :3, 3], fmt='%8.7f')
    np.savetxt(gt_t_filename, all_gt_T[:, :3, 3], fmt='%8.7f')
    plt.close(fig)


if __name__ == "__main__":
    train()
