import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from PIL import Image
from torchvision import transforms

from datasets.utils import extract_fp, rotation_to_euler


def pixel2cam(pt, fx, fy, cx, cy):
    pt[:, 0] = (pt[:, 0] - cx) / fx
    pt[:, 1] = (pt[:, 1] - cy) / fy
    return pt


def skew_symmetric_torch(w):
    """
    修复后的反对称矩阵构建 (支持批量处理)
    Args:
        w: (..., 3) 向量
    Returns:
        K: (..., 3, 3) 反对称矩阵
    """
    # 获取输入形状
    batch_shape = w.shape[:-1]

    # 提取分量
    w_x = w[..., 0]
    w_y = w[..., 1]
    w_z = w[..., 2]

    # 构建零矩阵
    zero = torch.zeros_like(w_x)

    # 构建反对称矩阵
    K = torch.stack(
        [
            torch.stack([zero, -w_z, w_y], dim=-1),
            torch.stack([w_z, zero, -w_x], dim=-1),
            torch.stack([-w_y, w_x, zero], dim=-1),
        ],
        dim=-2,
    )

    return K


def so3_to_rotation_matrix_torch(phi):
    """
    修复的 so3 -> 旋转矩阵转换，确保维度正确
    Args:
        phi: (batch_size, 3) so3向量
    Returns:
        R: (batch_size, 3, 3) 旋转矩阵
    """
    # 确保输入是二维的 [batch_size, 3]
    if phi.dim() == 1:
        phi = phi.unsqueeze(0)
    elif phi.dim() > 2:
        phi = phi.view(-1, 3)

    batch_size = phi.size(0)

    # 计算旋转角度
    theta = torch.norm(phi, dim=1, keepdim=True)  # [batch_size, 1]

    # 处理小角度情况
    small_angle_mask = theta < 1e-6
    theta_safe = torch.where(small_angle_mask, torch.ones_like(theta), theta)

    # 计算旋转轴
    axis = phi / theta_safe  # [batch_size, 3]

    # 提取轴分量
    ux, uy, uz = axis[:, 0], axis[:, 1], axis[:, 2]  # 每个都是 [batch_size]

    # 构建反对称矩阵的分量
    zero = torch.zeros_like(ux)

    # 直接构建旋转矩阵，避免复杂的重塑
    sin_theta = torch.sin(theta)  # [batch_size, 1]
    cos_theta = torch.cos(theta)  # [batch_size, 1]
    one_minus_cos = 1 - cos_theta  # [batch_size, 1]

    # 使用直接公式构建旋转矩阵，避免中间反对称矩阵
    R = torch.zeros(batch_size, 3, 3, device=phi.device, dtype=phi.dtype)

    # 对角线元素
    R[:, 0, 0] = cos_theta.squeeze() + ux * ux * one_minus_cos.squeeze()
    R[:, 1, 1] = cos_theta.squeeze() + uy * uy * one_minus_cos.squeeze()
    R[:, 2, 2] = cos_theta.squeeze() + uz * uz * one_minus_cos.squeeze()

    # 非对角线元素
    R[:, 0, 1] = ux * uy * one_minus_cos.squeeze() - uz * sin_theta.squeeze()
    R[:, 0, 2] = ux * uz * one_minus_cos.squeeze() + uy * sin_theta.squeeze()
    R[:, 1, 0] = uy * ux * one_minus_cos.squeeze() + uz * sin_theta.squeeze()
    R[:, 1, 2] = uy * uz * one_minus_cos.squeeze() - ux * sin_theta.squeeze()
    R[:, 2, 0] = uz * ux * one_minus_cos.squeeze() - uy * sin_theta.squeeze()
    R[:, 2, 1] = uz * uy * one_minus_cos.squeeze() + ux * sin_theta.squeeze()

    # 处理小角度情况的近似
    small_angle_indices = small_angle_mask.squeeze()
    if small_angle_indices.any():
        # 小角度近似: R ≈ I + [phi]×
        R_small = (
            torch.eye(3, device=phi.device, dtype=phi.dtype)
            .unsqueeze(0)
            .repeat(batch_size, 1, 1)
        )
        R_small[:, 0, 1] = -phi[:, 2]
        R_small[:, 0, 2] = phi[:, 1]
        R_small[:, 1, 0] = phi[:, 2]
        R_small[:, 1, 2] = -phi[:, 0]
        R_small[:, 2, 0] = -phi[:, 1]
        R_small[:, 2, 1] = phi[:, 0]

        # 只替换小角度的部分
        R[small_angle_indices] = R_small[small_angle_indices]

    return R


def rotation_matrix_to_so3_robust(R):
    """
    更稳健的旋转矩阵到 so(3) 转换
    """
    # 计算旋转角度
    cos_theta = (np.trace(R) - 1) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # 零旋转
        return np.zeros(3)

    # 使用更稳定的方法提取旋转轴
    # R - R^T = 2sin(θ)[v]×
    skew_matrix = R - R.T

    # 提取向量 [v_x, v_y, v_z]
    v_x = skew_matrix[2, 1]
    v_y = skew_matrix[0, 2]
    v_z = skew_matrix[1, 0]

    sin_theta = np.sin(theta)

    # 避免除零错误（虽然theta≠0，但sinθ可能很小）
    if abs(sin_theta) < 1e-10:
        # 使用泰勒展开近似
        axis = np.array([v_x, v_y, v_z]) / 2.0
    else:
        axis = np.array([v_x, v_y, v_z]) / (2 * sin_theta)

    # 归一化轴向量
    axis_norm = np.linalg.norm(axis)
    if axis_norm > 1e-10:
        axis = axis / axis_norm

    return theta * axis


def rotation_matrix_to_so3_torch(R):
    """
    可导的旋转矩阵 -> so3 转换 (PyTorch版本)
    Args:
        R: (..., 3, 3) 旋转矩阵
    Returns:
        phi: (..., 3) so3向量
    """
    # 计算旋转角度
    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(-1)
    cos_theta = (trace_R - 1) / 2.0

    # 确保数值稳定性
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)
    theta = torch.acos(cos_theta)

    # 处理小角度情况 (θ ≈ 0)
    small_angle = theta < 1e-6
    zero_vector = torch.zeros_like(R[..., 0, :])

    # 计算反对称部分
    skew_sym = (R - R.transpose(-1, -2)) / 2.0

    # 提取旋转轴向量
    v_x = skew_sym[..., 2, 1]
    v_y = skew_sym[..., 0, 2]
    v_z = skew_sym[..., 1, 0]
    axis_vec = torch.stack([v_x, v_y, v_z], dim=-1)

    # 计算 sinθ
    sin_theta = torch.sin(theta)

    # 避免除零错误
    sin_theta_safe = torch.where(
        sin_theta < 1e-6, torch.ones_like(sin_theta), sin_theta
    )
    axis = axis_vec / sin_theta_safe.unsqueeze(-1)

    # 归一化轴向量
    axis_norm = torch.norm(axis, dim=-1, keepdim=True)
    axis = axis / torch.where(axis_norm < 1e-6, torch.ones_like(axis_norm), axis_norm)

    # 组合结果
    phi = theta.unsqueeze(-1) * axis

    # 处理小角度情况
    phi = torch.where(small_angle.unsqueeze(-1), zero_vector, phi)

    return phi


class KITTI(torch.utils.data.Dataset):
    """
    Dataloader for KITTI Visual Odometry Dataset
        http://www.cvlibs.net/datasets/kitti/eval_odometry.php

    Arguments:
        data_path {str}: path to data sequences
        gt_path {str}: path to poses
    """

    def __init__(
        self,
        data_path=r"data/sequences_jpg",
        gt_path=r"data/poses",
        camera_id="2",
        # sequences=["00", "02", "08", "09"],
        sequences=["00"],
        window_size=3,
        overlap=1,
        read_poses=True,
        transform=None,
        train=False,
    ):

        self.data_path = data_path
        self.gt_path = gt_path
        self.camera_id = camera_id
        self.frame_id = 0
        self.read_poses = read_poses
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform

        self.train = train
        # KITTI normalization
        self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
        self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
        self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
        self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

        # define sequence for training, test and val
        self.sequences = sequences

        # read frames list and ground truths
        frames, seqs = self.read_frames()
        gt = self.read_gt()
        self.cam_params = {}

        # create dataframe with frames and ground truths
        data = pd.DataFrame({"gt": gt})
        data = data["gt"].apply(pd.Series)
        data["frames"] = frames
        data["sequence"] = seqs
        self.data = data
        self.read_intrinsics_param()
        self.K = torch.Tensor(
            [
                [self.cam_params["fx"], 0.0, self.cam_params["cx"]],
                [0.0, self.cam_params["fy"], self.cam_params["cy"]],
                [0.0, 0.0, 1.0],
            ]
        )
        # print(self.cam_params)
        self.windowed_data = self.create_windowed_dataframe(data)
        if not os.path.exists("fp.pickle"):
            self.generate_fp()
        self.fp = None
        with open("fp.pickle", "rb") as p:
            self.fp = pickle.load(p)

    def generate_fp(self):
        id_dict = {}
        for i in tqdm.tqdm(self.windowed_data["w_idx"].unique()):
            fp_dict = {}
            data = self.windowed_data.loc[self.windowed_data["w_idx"] == i]
            frame = data["frames"].values
            cvimage1 = cv2.imread(frame[0], cv2.IMREAD_GRAYSCALE)
            cvimage2 = cv2.imread(frame[1], cv2.IMREAD_GRAYSCALE)
            pt1, pt2 = extract_fp(cvimage1, cvimage2)
            pt1 = pixel2cam(
                pt1,
                self.cam_params["fx"],
                self.cam_params["fy"],
                self.cam_params["cx"],
                self.cam_params["cy"],
            )
            pt2 = pixel2cam(
                pt2,
                self.cam_params["fx"],
                self.cam_params["fy"],
                self.cam_params["cx"],
                self.cam_params["cy"],
            )
            fp_dict["pt1"] = pt1
            fp_dict["pt2"] = pt2
            id_dict[i] = fp_dict
        with open("fp.pickle", "wb") as p:
            pickle.dump(id_dict, p, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.windowed_data["w_idx"].unique())

    def __getitem__(self, idx):
        """
        Returns:
            frame {ndarray}: image frame at index self.frame_id
            pose {list}: list containing the ground truth pose [x, y, z]
            frame_id {int}: integer representing the frame index
        """
        # get data of corresponding window index
        data = self.windowed_data.loc[self.windowed_data["w_idx"] == idx, :]

        fp = self.fp[idx]
        pt1 = fp["pt1"]
        pt2 = fp["pt2"]

        # Read frames as grayscale
        frames = data["frames"].values
        imgs = []
        for fname in frames:
            img = Image.open(fname).convert("RGB")
            # pre processing
            img = self.transform(img)
            img = img.unsqueeze(0)
            imgs.append(img)
        imgs = np.concatenate(imgs, axis=0)
        imgs = np.asarray(imgs)
        # T C H W -> C T H W.
        imgs = imgs.transpose(1, 0, 2, 3)

        # Read ground truth [window_size-1 x 6]
        gt_poses = data.loc[:, [i for i in range(12)]].values
        y = []
        for gt_idx, gt in enumerate(gt_poses):

            # homogeneous pose matrix [4 x 4]
            pose = np.vstack([np.reshape(gt, (3, 4)), [[0.0, 0.0, 0.0, 1.0]]])

            # compute relative pose from frame1 to frame2
            if gt_idx > 0:
                pose_wrt_prev = np.dot(np.linalg.inv(pose_prev), pose)
                R = pose_wrt_prev[:3, :3]
                t = pose_wrt_prev[:3, 3]

                # Euler parameterization (rotations as Euler angles)
                angles = rotation_to_euler(R, seq="zyx")
                # phi = rotation_matrix_to_so3_robust(R)

                # normalization
                # angles = (np.asarray(angles) - self.mean_angles) / self.std_angles
                # t = (np.asarray(t) - self.mean_t) / self.std_t

                # concatenate angles and translation
                y.append(list(angles) + list(t))

            pose_prev = pose

        y = np.asarray(y)  # discard first value
        # y = y.flatten()

        # pt1 = torch.Tensor([])
        # pt2 = torch.Tensor([])

        return imgs, pt1, pt2, y

    def read_intrinsics_param(self):
        """
        Reads camera intrinsics parameters

        Returns:
            cam_params {dict}: dictionary with focal lenght and principal point
        """
        # calib_file = os.path.join(self.data_path, self.sequence, "calib.txt")
        calib_file = os.path.join(self.data_path, "00", "calib.txt")
        with open(calib_file, "r") as f:
            lines = f.readlines()
            line = lines[int(self.camera_id)].strip().split()
            [fx, cx, fy, cy] = [
                float(line[1]),
                float(line[3]),
                float(line[6]),
                float(line[7]),
            ]

            # focal length of camera
            self.cam_params["fx"] = fx
            self.cam_params["fy"] = fy
            # principal point (optical center)
            self.cam_params["cx"] = cx
            self.cam_params["cy"] = cy

    def read_frames(self):
        # Get frames list
        frames = []
        seqs = []
        for sequence in self.sequences:
            frames_dir = os.path.join(
                self.data_path, sequence, "image_{}".format(self.camera_id), "*.png"
            )
            frames_seq = sorted(glob.glob(frames_dir))
            frames = frames + frames_seq
            seqs = seqs + [sequence] * len(frames_seq)
        return frames, seqs

    def read_gt(self):
        # Read ground truth
        if self.read_gt:
            gt = []
            for sequence in self.sequences:
                with open(os.path.join(self.gt_path, sequence + ".txt")) as f:
                    lines = f.readlines()

                # convert poses to float
                for line_idx, line in enumerate(lines):
                    line = line.strip().split()
                    line = [float(x) for x in line]
                    gt.append(line)

        else:  # test data (sequences 11-21)
            gt = None

        return gt

    def create_windowed_dataframe(self, df):
        window_size = self.window_size
        overlap = self.overlap
        windowed_df = pd.DataFrame()
        w_idx = 0

        for sequence in df["sequence"].unique():
            seq_df = df.loc[df["sequence"] == sequence, :].reset_index(drop=True)
            row_idx = 0
            while row_idx + window_size <= len(seq_df):

                rows = seq_df.iloc[row_idx : (row_idx + window_size)].copy()
                rows["w_idx"] = len(rows) * [w_idx]  # add window index column
                row_idx = row_idx + window_size - overlap
                w_idx = w_idx + 1
                windowed_df = pd.concat([windowed_df, rows], ignore_index=True)
        windowed_df.reset_index(drop=True)
        return windowed_df


if __name__ == "__main__":

    # Create dataloader
    preprocess = transforms.Compose(
        [
            transforms.Resize((192, 640)),
            # transforms.Resize((224, 678)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.34721234, 0.36705238, 0.36066107],
                std=[0.30737526, 0.31515116, 0.32020183],
            ),
        ]
    )

    data = KITTI(transform=preprocess, sequences=["04"], window_size=3, overlap=2)
    test_loader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)

    timings = np.zeros((len(test_loader), 1))
    for i in range(len(test_loader)):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()
        imgs, gt = data[i]
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[i] = curr_time

    mean_syn = np.sum(timings) / len(test_loader)
    std_syn = np.std(timings)
    print("Mean pre-proc time: ", mean_syn)
    print("Std time: ", std_syn)

# idx = 500
# imgs, gt = data[idx]
# print("imgs.shape: ", imgs.shape)
# print("gt.shape: ", gt.shape)

# img = np.moveaxis(imgs[0, :, :, :], 0, -1)

# # post processing
# channelwise_mean = [0.34721234, 0.36705238, 0.36066107]
# channelwise_std = [0.30737526, 0.31515116, 0.32020183]
# img[:, :, 0] = img[:, :, 0] * channelwise_std[0] + channelwise_mean[0]
# img[:, :, 1] = img[:, :, 1] * channelwise_std[1] + channelwise_mean[1]
# img[:, :, 2] = img[:, :, 2] * channelwise_std[2] + channelwise_mean[2]

# plt.figure()
# plt.imshow((img * 255).astype(int));
