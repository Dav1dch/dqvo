import json
import os
import pickle

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from build_model import build_model
from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation_torch

from tensorboard import program


torch.manual_seed(2023)


def vector_to_skew_symmetric(v):
    """
    将 (B, 3) 的向量转换为 (B, 3, 3) 的反对称矩阵
    """
    B = v.shape[0]
    # 提取分量
    v1, v2, v3 = v[:, 0], v[:, 1], v[:, 2]
    # 构建反对称矩阵
    zeros = torch.zeros(B, device=v.device)
    skew = torch.stack(
        [
            torch.stack([zeros, -v3, v2], dim=1),
            torch.stack([v3, zeros, -v1], dim=1),
            torch.stack([-v2, v1, zeros], dim=1),
        ],
        dim=1,
    )
    return skew  # 形状 (B, 3, 3)


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


def val_epoch(model, val_loader, criterion, args):
    epoch_loss = 0
    with tqdm(val_loader, unit="batch", dynamic_ncols=True) as tepoch:
        # for images, gt, g_gt in tepoch:
        for images, pt1, pt2, gt in tepoch:
            tepoch.set_description(f"Validating ")
            # for batch_idx, (images, odom) in enumerate(train_loader):
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())

            # compute loss
            loss = compute_loss(estimated_pose, gt, criterion, args)

            epoch_loss += loss.item()
            tepoch.set_postfix(val_loss=loss.item())

    return epoch_loss / len(val_loader)


import functools


def denormailzation(angles, t, normalizations):
    angles = (
        torch.multiply(angles, normalizations["mean_angles"])
        + normalizations["std_angles"]
    )
    t = torch.multiply(t, normalizations["mean_t"]) + normalizations["std_t"]
    return angles, t


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch,
    tensorboard_writer,
    args,
    normalizations,
):
    epoch_loss = 0
    iter = (epoch - 1) * len(train_loader) + 1
    iter = 1
    e_loss = 0

    with tqdm(train_loader, unit="batch", dynamic_ncols=True) as tepoch:
        for images, pt1, pt2, gt in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            if torch.cuda.is_available():
                images, pt1, pt2, gt = images.cuda(), pt1.cuda(), pt2.cuda(), gt.cuda()

            # predict pose
            estimated_pose = model(images.float())
            # estimated_angles = estimated_pose[:, :, :3]
            # estimated_t = estimated_pose[:, :, 3:]
            # estimated_angles, estimated_t = denormailzation(
            #     estimated_angles, estimated_t, normalizations
            # )
            # ones = torch.ones(pt1.shape[0], pt1.shape[1], 1).cuda()
            # pt1 = torch.cat([pt1, ones], dim=-1)
            # pt2 = torch.cat([pt2, ones], dim=-1)
            # R = euler_to_rotation_matrix(estimated_angles.squeeze(1))
            # t_x = vector_to_skew_symmetric(estimated_t.squeeze(1))
            #
            # E = torch.bmm(t_x, R)
            # term = torch.bmm(
            #     pt2.view(pt2.shape[0] * pt2.shape[1], 1, 3),
            #     E.unsqueeze(1)
            #     .expand(pt2.shape[0], pt2.shape[1], 3, 3)
            #     .reshape(pt2.shape[0] * pt2.shape[1], 3, 3),
            # )
            # epipolar_loss = torch.bmm(
            #     term,
            #     pt1.view(pt1.shape[0] * pt1.shape[1], 3, 1),
            # )
            # print(epipolar_loss.abs().mean())
            loss = (
                compute_loss(estimated_pose, gt, criterion, args)
                # + epipolar_loss.abs().mean()
            )
            # e_loss += epipolar_loss.abs().mean().item()

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # tepoch.set_postfix(loss=loss.item())
            tepoch.set_postfix(loss=epoch_loss / iter)
            # tepoch.set_postfix({"loss": epoch_loss / iter, "e_loss": e_loss / iter})

            # log tensorboard
            # tensorboard_writer.add_scalar('training_loss', loss.item(), iter)
            iter += 1
    return epoch_loss / len(train_loader)


def train(
    model, train_loader, val_loader, criterion, optimizer, tensorboard_writer, args
):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]
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

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
    for epoch in range(args["epoch_init"], epochs):
        # training for one epoch
        model.train()
        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            tensorboard_writer,
            args,
            normalizations,
        )
        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
        }

        # validate model
        if val_loader and not epoch % 2:
            with torch.no_grad():
                model.eval()
                val_loss = val_epoch(model, val_loader, criterion, args)

            print(
                f"Epoch: {epoch} - loss: {train_loss:.4f} - val_loss: {val_loss:.4f} \n"
            )

            # save best mode
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            }
            if val_loss < best_val:
                print(
                    f"Saving new best model -- loss decreased from {best_val:.6f} to {val_loss:.6f} \n"
                )
                best_val = val_loss
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            # log validation loss in TensorBoard
            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        # save checkpoint every 20 epochs
        if not epoch % 20:
            torch.save(
                state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))
            )
        # save last checkpoint
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))

        # log loss in TensorBoard
        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def get_optimizer(params, args):
    method = args["optimizer"]

    # initialize the optimizer
    if method == "Adam":
        optimizer = optim.Adam(params, lr=args["lr"])
    elif method == "SGD":
        optimizer = optim.SGD(
            params,
            lr=args["lr"],
            momentum=args["momentum"],
            weight_decay=args["weight_decay"],
        )
    elif method == "RAdam":
        optimizer = optim.RAdam(params, lr=args["lr"])
    elif method == "Adagrad":
        optimizer = optim.Adagrad(
            params, lr=args["lr"], weight_decay=args["weight_decay"]
        )

    # load checkpoint
    if args["checkpoint"] is not None:
        checkpoint = torch.load(
            os.path.join(args["checkpoint_path"], args["checkpoint"])
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return optimizer


def compute_loss(y_hat, y, criterion, args):
    if args["weighted_loss"] == None:
        loss = criterion(y_hat, y.float())
    else:
        y = torch.reshape(y, (y.shape[0], args["window_size"] - 1, 6))
        gt_angles = y[:, :, :3].flatten()
        gt_translation = y[:, :, 3:].flatten()

        # predict pose
        y_hat = torch.reshape(y_hat, (y_hat.shape[0], args["window_size"] - 1, 6))
        estimated_angles = y_hat[:, :, :3].flatten()
        estimated_translation = y_hat[:, :, 3:].flatten()

        # compute custom loss
        k = args["weighted_loss"]
        loss_angles = k * criterion(estimated_angles, gt_angles.float())
        loss_translation = criterion(estimated_translation, gt_translation.float())
        loss = loss_angles + loss_translation
    return loss


if __name__ == "__main__":

    # set hyperparameters and configuration
    args = {
        "data_dir": "data",
        "bsize": 8,  # batch size
        "val_split": 0.1,  # percentage to use as validation data
        "window_size": 3,  # number of frames in window
        "overlap": 2,  # number of frames overlapped between windows
        "optimizer": "Adam",  # optimizer [Adam, SGD, Adagrad, RAdam]
        "lr": 1e-5,  # learning rate
        "momentum": 0.9,  # SGD momentum
        "weight_decay": 1e-4,  # SGD momentum
        "epoch": 100,  # train iters each timestep
        "weighted_loss": None,  # float to weight angles in loss function
        "pretrained_ViT": False,  # load weights from pre-trained ViT
        "checkpoint_path": "checkpoints/Exp30",  # path to save checkpoint
        # "checkpoint": "checkpoint_best.pth",  # checkpoint
        "checkpoint": None,  # checkpoint
    }

    # tiny  - patch_size=16, embed_dim=192, depth=12, num_heads=3
    # small - patch_size=16, embed_dim=384, depth=12, num_heads=6
    # base  - patch_size=16, embed_dim=768, depth=12, num_heads=12
    model_params = {
        "dim": 384,
        # "image_size": (224, 678),  # (192, 640),
        "image_size": (192, 640),  # (192, 640),
        "patch_size": 16,
        "attention_type": "divided_space_time",  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),  # 6 DoF for each frame
        "depth": 16,
        "heads": 3,
        "dim_head": 64,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "time_only": False,
    }
    args["model_params"] = model_params

    # create checkpoints folder
    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], "args.pkl"), "wb") as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], "args.txt"), "w") as f:
        f.write(json.dumps(args))

    # tensorboard writer
    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])
    # tb = program.TensorBoard()
    # tb.configure(
    #     argv=[
    #         None,
    #         "--logdir_spec",
    #         "Exp22:./checkpoints/Exp22",
    #         "--bind_all",
    #         "--port=6006",
    #     ]
    # )
    # url = tb.launch()
    # print(f"TensorBoard URL ：{url}")

    # preprocessing operation
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

    # train and val dataloader
    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")
    dataset = KITTI(
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess,
        train=True,
    )
    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(
        dataset, [len(dataset) - nb_val, nb_val]
    )  # generator=torch.Generator().manual_seed(2))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args["bsize"],
        num_workers=8,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        num_workers=8,
        shuffle=False,
    )

    # build and load model
    print("Building model...")
    model, args = build_model(args, model_params)

    n = sum([param.nelement() for param in model.parameters()])
    print(n)

    # loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = get_optimizer(model.parameters(), args)

    # train network
    print(20 * "--" + " Training " + 20 * "--")
    train(
        model, train_loader, val_loader, criterion, optimizer, TensorBoardWriter, args
    )
