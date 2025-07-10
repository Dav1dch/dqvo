from pickletools import optimize

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from build_model import build_model
from datasets.utils import rotation_to_euler

mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

K = [
    7.188560000000e02,
    0.000000000000e00,
    6.071928000000e02,
    0.000000000000e00,
    7.188560000000e02,
    1.852157000000e02,
    0.000000000000e00,
    0.000000000000e00,
    1.000000000000e00,
]


pose1 = [
    1.000000e00,
    9.043680e-12,
    2.326809e-11,
    5.551115e-17,
    9.043683e-12,
    1.000000e00,
    2.392370e-10,
    3.330669e-16,
    2.326810e-11,
    2.392370e-10,
    9.999999e-01,
    -4.440892e-16,
]
pose2 = [
    9.999978e-01,
    5.272628e-04,
    -2.066935e-03,
    -4.690294e-02,
    -5.296506e-04,
    9.999992e-01,
    -1.154865e-03,
    -2.839928e-02,
    2.066324e-03,
    1.155958e-03,
    9.999971e-01,
    8.586941e-01,
]
pose1 = np.vstack([np.reshape(pose1, (3, 4)), [[0.0, 0.0, 0.0, 1.0]]])
pose2 = np.vstack([np.reshape(pose2, (3, 4)), [[0.0, 0.0, 0.0, 1.0]]])
pose_wrt_prev = np.dot(np.linalg.inv(pose1), pose2)
R = pose_wrt_prev[:3, :3]
t = pose_wrt_prev[:3, 3]

# Euler parameterization (rotations as Euler angles)
angles = rotation_to_euler(R, seq="zyx")

# normalization
# gt_angles = (np.asarray(angles) - mean_angles) / std_angles
# gt_t = (np.asarray(t) - mean_t) / std_t
gt_angles = angles
gt_t = t

gt_pose = torch.Tensor(list(gt_angles) + list(gt_t)).cuda()


def pixel2cam(pt, K):
    pt[:, 0] = (pt[:, 0] - K[0, 2]) / K[0, 0]
    pt[:, 1] = (pt[:, 1] - K[1, 2]) / K[1, 1]
    return pt


def euler_to_rotation_matrix(
    euler_angles: torch.Tensor, order: str = "ZYX"
) -> torch.Tensor:
    """
    将欧拉角转换为旋转矩阵（支持批量输入）

    Args:
        euler_angles: 形状为 (batch_size, 3) 的张量，每个元素为 [ψ, θ, φ]（弧度）
        order: 旋转顺序，默认为 'ZYX'

    Returns:
        旋转矩阵张量，形状为 (batch_size, 3, 3)
    """
    batch_size = euler_angles.shape[0]
    psi, theta, phi = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]

    # 预计算sin和cos
    sin_psi, cos_psi = torch.sin(psi), torch.cos(psi)
    sin_theta, cos_theta = torch.sin(theta), torch.cos(theta)
    sin_phi, cos_phi = torch.sin(phi), torch.cos(phi)

    # 构造各轴旋转矩阵（批量形式）
    Rz = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
    Rz[:, 0, 0] = cos_psi
    Rz[:, 0, 1] = -sin_psi
    Rz[:, 1, 0] = sin_psi
    Rz[:, 1, 1] = cos_psi
    Rz[:, 2, 2] = 1.0

    Ry = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
    Ry[:, 0, 0] = cos_theta
    Ry[:, 0, 2] = sin_theta
    Ry[:, 1, 1] = 1.0
    Ry[:, 2, 0] = -sin_theta
    Ry[:, 2, 2] = cos_theta

    Rx = torch.zeros((batch_size, 3, 3), device=euler_angles.device)
    Rx[:, 0, 0] = 1.0
    Rx[:, 1, 1] = cos_phi
    Rx[:, 1, 2] = -sin_phi
    Rx[:, 2, 1] = sin_phi
    Rx[:, 2, 2] = cos_phi

    # 按顺序矩阵乘法（ZYX）
    if order == "ZYX":
        R = torch.bmm(Rz, torch.bmm(Ry, Rx))
    elif order == "XYZ":
        # 其他顺序需调整矩阵乘法顺序（如XYZ）
        R = torch.bmm(Rx, torch.bmm(Ry, Rz))
    else:
        raise ValueError(f"Unsupported rotation order: {order}")

    return R


def extract_fp(cvimage1, cvimage2):
    orb = cv2.ORB_create()

    k1, d1 = orb.detectAndCompute(cvimage1, None)
    k2, d2 = orb.detectAndCompute(cvimage2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(d1, d2)

    matches = sorted(matches, key=lambda x: x.distance)

    pt1 = []
    pt2 = []

    for m in matches[:50]:
        pt1.append([list(map(round, list(k1[m.queryIdx].pt)))])
        pt2.append([list(map(round, list(k2[m.trainIdx].pt)))])
    pt1 = torch.Tensor(pt1).float().squeeze(1)
    pt2 = torch.Tensor(pt2).float().squeeze(1)
    return pt1, pt2


def denormailzation(angles, t, normalizations):
    angles = (
        torch.multiply(angles, normalizations["mean_angles"])
        + normalizations["std_angles"]
    )
    t = torch.multiply(t, normalizations["mean_t"]) + normalizations["std_t"]
    return angles, t


def main():
    global K, gt_pose
    normalizations = {}
    # self.mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
    # self.std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
    # self.mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
    # self.std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])
    normalizations["mean_angles"] = torch.Tensor(
        [1.7061e-5, 9.5582e-4, -5.5258e-5]
    ).cuda()
    normalizations["std_angles"] = torch.Tensor(
        [2.8256e-3, 1.7771e-2, 3.2326e-3]
    ).cuda()
    normalizations["mean_t"] = torch.Tensor([-8.6736e-5, -1.6038e-2, 9.0033e-1]).cuda()
    normalizations["std_t"] = torch.Tensor([2.5584e-2, 1.8545e-2, 3.0352e-1]).cuda()
    args = {
        "data_dir": "data",
        "bsize": 4,  # batch size
        "val_split": 0.1,  # percentage to use as validation data
        "window_size": 2,  # number of frames in window
        "overlap": 1,  # number of frames overlapped between windows
        "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 1e-4,  # SGD momentum
        "epoch": 300,  # train iters each timestep
        "weighted_loss": None,  # float to weight angles in loss function
        "pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints/Exp18",  # path to save checkpoint
        # "checkpoint": "checkpoint_best.pth",  # checkpoint
        "checkpoint": None,  # checkpoint
    }

    # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
    # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
    # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
        "dim": 384,
        "image_size": (192, 640),  # (192, 640),
        "patch_size": 16,
        "attention_type": "divided_space_time",  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 12,
        "heads": 3,
        "dim_head": 64,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "time_only": False,
    }
    args["model_params"] = model_params

    model, args = build_model(args, model_params)
    model = model.cuda()

    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.MSELoss()
    preprocess = transforms.Compose(
        [
            transforms.Resize((model_params["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.34721234, 0.36705238, 0.36066107],
                std=[0.30737526, 0.31515116, 0.32020183],
            ),
        ]
    )
    img1_dir = "./data/sequences_jpg/00/image_2/000000.png"
    img2_dir = "./data/sequences_jpg/00/image_2/000001.png"
    image1 = preprocess(Image.open(img1_dir).convert("RGB"))
    image2 = preprocess(Image.open(img2_dir).convert("RGB"))
    input_images = (
        torch.stack([image1, image2]).permute((1, 0, 2, 3)).unsqueeze(0).cuda()
    )
    output = model(input_images)
    print(euler_to_rotation_matrix(output[0][:3]))

    cvimage1 = cv2.imread(img1_dir, cv2.IMREAD_GRAYSCALE)
    cvimage2 = cv2.imread(img2_dir, cv2.IMREAD_GRAYSCALE)
    pt1, pt2 = extract_fp(cvimage1, cvimage2)

    # T = [
    #     9.999978e-01,
    #     5.272628e-04,
    #     -2.066935e-03,
    #     -4.690294e-02,
    #     -5.296506e-04,
    #     9.999992e-01,
    #     -1.154865e-03,
    #     -2.839928e-02,
    #     2.066324e-03,
    #     1.155958e-03,
    #     9.999971e-01,
    #     8.586941e-01,
    # ]
    # T = torch.Tensor(T).view((3, 4))

    # R = T[:3, :3]
    # t = T[:, 3:].view((3))
    # t = torch.Tensor(t)
    # R = euler_to_rotation_matrix(torch.Tensor(angles).unsqueeze(0)).squeeze(0)

    # t_x = torch.Tensor([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])

    K = torch.Tensor(K).view((3, 3)).cuda()

    pt1 = pixel2cam(pt1.cuda(), K)
    pt2 = pixel2cam(pt2.cuda(), K)
    ones = torch.ones((pt1.shape[0], 1)).cuda()
    pt1 = torch.hstack((pt1, ones)).unsqueeze(-1)
    pt2 = torch.hstack((pt2, ones)).unsqueeze(1)
    mean_angles = np.array([1.7061e-5, 9.5582e-4, -5.5258e-5])
    std_angles = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
    mean_t = np.array([-8.6736e-5, -1.6038e-2, 9.0033e-1])
    std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])

    gt_pose = gt_pose.unsqueeze(0)
    # gt_pose = gt_pose.unsqueeze(0).cpu().numpy()
    # print(gt_pose)
    # angles = gt_pose[:, :3].squeeze(0)
    # t = gt_pose[:, 3:].squeeze(0)
    # angles = (np.asarray(angles) - mean_angles) / std_angles
    # t = (np.asarray(t) - mean_t) / std_t
    # gt_pose = torch.Tensor(angles.tolist() + t.tolist()).cuda()
    # print(gt_pose)

    losses = []

    for i in range(300):
        output = model(input_images).squeeze(0)
        loss1 = criterion(output, gt_pose)
        t = output[:, 3:]
        angles = output[:, :3]

        # angles, t = denormailzation(angles, t, normalizations)
        R = euler_to_rotation_matrix(angles).squeeze()
        t_x = torch.Tensor(
            [[0, -t[0][2], t[0][1]], [t[0][2], 0, -t[0][0]], [-t[0][1], t[0][0], 0]]
        ).cuda()
        E = torch.mm(t_x, R)
        term = torch.bmm(pt2, E.unsqueeze(0).expand(50, 3, 3))
        result = torch.bmm(term, pt1)
        loss2 = result.abs().mean()
        losses.append(loss1.item())
        print(loss2)
        loss = loss2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    E = torch.mm(t_x, R)
    term = torch.bmm(pt2, E.unsqueeze(0).expand(50, 3, 3))
    result = torch.bmm(term, pt1)
    mean_error = result.abs().mean().item()
    print(mean_error)
    # groundtruth
    # 9.999978e-01 5.272628e-04 -2.066935e-03 -4.690294e-02 -5.296506e-04 9.999992e-01 -1.154865e-03 -2.839928e-02 2.066324e-03 1.155958e-03 9.999971e-01 8.586941e-01

    x = np.arange(0, len(losses))
    plt.figure(figsize=(10, 6))
    plt.plot(x, losses)
    plt.show()

    # intrinsic
    # P2: 7.188560000000e+02 0.000000000000e+00 6.071928000000e+02
    # 0.000000000000e+00 7.188560000000e+02 1.852157000000e+02
    # 0.000000000000e+00 0.000000000000e+00 1.000000000000e+00

    # Display the matched image
    # cv2.imshow("ORB Matches", matched_image)
    # while cv2.waitKey(33) != ord("a"):
    #     continue
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
