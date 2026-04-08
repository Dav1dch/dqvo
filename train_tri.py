import json
import os
import pickle
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
dqvo_root = os.path.dirname(script_dir)
dual_quat_vo_dir = os.path.join(dqvo_root, "DualQuaternionVo")

new_path = [p for p in sys.path if "dqvo" not in p and "DualQuaternionVo" not in p]
new_path = [p for p in new_path if p != ""]

final_path = [script_dir]
final_path.append(os.path.join(script_dir, "datasets"))
final_path.append(dual_quat_vo_dir)
final_path.append(os.path.join(script_dir, "timesformer"))
final_path.extend(new_path)

sys.path = final_path

import torch
import torch.optim as optim
from torch.utils.data import random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from build_model import build_model
from datasets.kitti import KITTI
from datasets.utils import euler_to_rotation_torch, kitti_pose_to_dual_quaternion
from timesformer.models.losses import dual_quaternion_loss, quaternion_loss_weighted
from losses.triangulation_loss import (
    TriangulationLoss,
    ResidualTriangulationLoss,
    DualQuaternionVOLoss,
)

torch.manual_seed(2023)


def val_epoch(
    model, val_loader, criterion, tri_loss, args, cam_K, residual_tri_loss=None
):
    epoch_loss = 0
    epoch_geo_loss = 0
    epoch_tri_loss = 0
    epoch_res_tri_loss = 0
    num_batches = 0

    with tqdm(val_loader, unit="batch", dynamic_ncols=True) as tepoch:
        for images, pt1, pt2, gt, n_valid in tepoch:
            tepoch.set_description(f"Validating ")
            if torch.cuda.is_available():
                images, pt1, pt2, gt = images.cuda(), pt1.cuda(), pt2.cuda(), gt.cuda()

            estimated_pose = model(images.float())

            loss, loss_dict = compute_loss(
                estimated_pose,
                gt,
                criterion,
                tri_loss,
                args,
                pt1,
                pt2,
                cam_K,
                n_valid,
                residual_tri_loss_fn=residual_tri_loss,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[Val] Warning: NaN/Inf loss detected")
                print(f"  loss_dict: {loss_dict}")
                print(f"  geo_loss: {loss_dict.get('geo', 'N/A')}")
                print(f"  tri_loss: {loss_dict.get('tri', 'N/A')}")
                print(f"  res_tri_loss: {loss_dict.get('res_tri', 'N/A')}")
                continue
            epoch_loss += loss.item()
            epoch_geo_loss += loss_dict.get("geo", 0)
            epoch_tri_loss += loss_dict.get("tri", 0)
            epoch_res_tri_loss += loss_dict.get("res_tri", 0)
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_geo = epoch_geo_loss / num_batches
            avg_tri = epoch_tri_loss / num_batches
            avg_res_tri = epoch_res_tri_loss / num_batches

            tepoch.set_postfix(
                loss=avg_loss, geo=avg_geo, tri=avg_tri, res_tri=avg_res_tri
            )

    return {
        "total": epoch_loss / num_batches if num_batches > 0 else 0,
        "geo": epoch_geo_loss / num_batches if num_batches > 0 else 0,
        "tri": epoch_tri_loss / num_batches if num_batches > 0 else 0,
        "res_tri": epoch_res_tri_loss / num_batches if num_batches > 0 else 0,
    }


def train_epoch(
    model,
    train_loader,
    criterion,
    optimizer,
    epoch,
    tensorboard_writer,
    args,
    tri_loss,
    cam_K,
    residual_tri_loss=None,
):
    epoch_loss = 0
    epoch_geo_loss = 0
    epoch_tri_loss = 0
    epoch_res_tri_loss = 0
    num_batches = 0
    iter = (epoch - 1) * len(train_loader) + 1
    last_batch_loss = None
    last_geo_loss = None
    last_tri_loss = None
    last_res_tri_loss = None

    with tqdm(train_loader, unit="batch", dynamic_ncols=True) as tepoch:
        for images, pt1, pt2, gt, n_valid in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            if torch.cuda.is_available():
                images, pt1, pt2, gt = images.cuda(), pt1.cuda(), pt2.cuda(), gt.cuda()

            if torch.isnan(images).any():
                print(f"\n[ERROR] NaN in input images, STOPPING!")
                print(f"  batch_idx: {num_batches}")
                print(f"  images shape: {images.shape}")
                print(f"  last batch loss: {last_batch_loss}")
                sys.exit(1)

            estimated_pose = model(images.float())

            # Check for large values that could lead to NaN
            pose_abs_max = torch.abs(estimated_pose).max().item()
            if pose_abs_max > 100:
                print(
                    f"\n[WARNING] Large model output! abs_max={pose_abs_max:.2f}, batch_idx={num_batches}"
                )

            if torch.isnan(estimated_pose).any() or torch.isinf(estimated_pose).any():
                print(f"\n[ERROR] NaN/Inf in model output, STOPPING!")
                print(f"  batch_idx: {num_batches}")
                print(f"  images shape: {images.shape}, dtype: {images.dtype}")
                print(
                    f"  images range: [{images.min().item():.4f}, {images.max().item():.4f}]"
                )
                print(f"  y_hat shape: {estimated_pose.shape}")
                print(f"  y_hat sample: {estimated_pose[0,:]}")
                print(f"  last batch loss: {last_batch_loss}")
                print(f"  last geo loss: {last_geo_loss}")
                print(f"  last tri loss: {last_tri_loss}")
                sys.exit(1)

            loss, loss_dict = compute_loss(
                estimated_pose,
                gt,
                criterion,
                tri_loss,
                args,
                pt1,
                pt2,
                cam_K,
                n_valid,
                residual_tri_loss_fn=residual_tri_loss,
            )

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n[ERROR] NaN/Inf loss detected, STOPPING!")
                print(f"  batch_idx: {num_batches}")
                print(f"  current loss_dict: {loss_dict}")
                print(f"  current geo_loss: {loss_dict.get('geo', 'N/A')}")
                print(f"  current tri_loss: {loss_dict.get('tri', 'N/A')}")
                print(
                    f"  y_hat sample: {estimated_pose[0,:10] if estimated_pose.numel() > 0 else 'empty'}"
                )
                print(f"  y sample: {gt[0,:10] if gt.numel() > 0 else 'empty'}")
                print(f"  === LAST BATCH (before NaN) ===")
                print(f"  last_batch_loss: {last_batch_loss}")
                print(f"  last_geo_loss: {last_geo_loss}")
                print(f"  last_tri_loss: {last_tri_loss}")
                sys.exit(1)

            optimizer.zero_grad()
            loss.backward()

            # Check for gradient explosion before stepping
            total_grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_grad_norm += p.grad.data.norm(2).item() ** 2
            total_grad_norm = total_grad_norm**0.5

            if total_grad_norm > 100.0:
                # print(
                #     f"\n[WARNING] Gradient explosion detected! grad_norm={total_grad_norm:.2f}, clipping..."
                # )
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            last_batch_loss = loss.item()
            last_geo_loss = loss_dict.get("geo", 0)
            last_tri_loss = loss_dict.get("tri", 0)
            last_res_tri_loss = loss_dict.get("res_tri", 0)
            epoch_loss += loss.item()
            epoch_geo_loss += loss_dict.get("geo", 0)
            epoch_tri_loss += loss_dict.get("tri", 0)
            epoch_res_tri_loss += loss_dict.get("res_tri", 0)
            num_batches += 1

            avg_loss = epoch_loss / num_batches
            avg_geo = epoch_geo_loss / num_batches
            avg_tri = epoch_tri_loss / num_batches
            avg_res_tri = epoch_res_tri_loss / num_batches

            tepoch.set_postfix(
                loss=avg_loss, geo=avg_geo, tri=avg_tri, res_tri=avg_res_tri
            )

    return {
        "total": epoch_loss / num_batches if num_batches > 0 else 0,
        "geo": epoch_geo_loss / num_batches if num_batches > 0 else 0,
        "tri": epoch_tri_loss / num_batches if num_batches > 0 else 0,
        "res_tri": epoch_res_tri_loss / num_batches if num_batches > 0 else 0,
    }


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    tensorboard_writer,
    args,
    tri_loss,
    cam_K,
    residual_tri_loss=None,
):
    checkpoint_path = args["checkpoint_path"]
    epochs = args["epoch"]
    best_val = args["best_val"]

    for epoch in range(args["epoch_init"], epochs):
        model.train()
        train_loss_dict = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            tensorboard_writer,
            args,
            tri_loss,
            cam_K,
            residual_tri_loss,
        )

        train_loss = train_loss_dict["total"]

        state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val": best_val,
        }

        if val_loader and not epoch % 1:
            with torch.no_grad():
                model.eval()
                val_loss_dict = val_epoch(
                    model,
                    val_loader,
                    criterion,
                    tri_loss,
                    args,
                    cam_K,
                    residual_tri_loss,
                )

            val_loss = val_loss_dict["total"]

            print(
                f"Epoch: {epoch} - train_loss: {train_loss:.4f} (geo: {train_loss_dict['geo']:.4f}, tri: {train_loss_dict['tri']:.4f}) - "
                f"val_loss: {val_loss:.4f} (geo: {val_loss_dict['geo']:.4f}, tri: {val_loss_dict['tri']:.4f})\n"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val": best_val,
            }

            val_geo = val_loss_dict.get("geo", val_loss)
            val_res_tri = val_loss_dict.get("res_tri", 0)
            eval_metric = val_geo + val_res_tri

            if eval_metric < best_val:
                print(
                    f"Saving new best model -- eval_metric decreased from {best_val:.6f} to {eval_metric:.6f} "
                    f"(geo: {val_geo:.4f}, res_tri: {val_res_tri:.4f})\n"
                )
                best_val = eval_metric
                state["best_val"] = best_val
                torch.save(state, os.path.join(checkpoint_path, "checkpoint_best.pth"))

            tensorboard_writer.add_scalar("val_loss", val_loss, epoch)

        if not epoch % 20:
            torch.save(
                state, os.path.join(checkpoint_path, "checkpoint_e{}.pth".format(epoch))
            )
        torch.save(state, os.path.join(checkpoint_path, "checkpoint_last.pth"))

        tensorboard_writer.add_scalar("train_loss", train_loss, epoch)
    return


def get_optimizer(params, args):
    method = args["optimizer"]

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

    if args["checkpoint"] is not None and args["checkpoint"] != "base.pth":
        checkpoint = torch.load(
            os.path.join(args["checkpoint_path"], args["checkpoint"])
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return optimizer


def compute_loss(
    y_hat,
    y,
    criterion,
    tri_loss_fn,
    args,
    pt1,
    pt2,
    K,
    n_valid=None,
    dq_loss_fn=None,
    residual_tri_loss_fn=None,
):
    batch_size = y.shape[0]
    window_size = args["window_size"]
    device = y_hat.device

    if dq_loss_fn is None:
        dq_loss_fn = dual_quaternion_loss

    y_flat = y.reshape(-1, 7)
    y_hat_flat = y_hat.reshape(-1, 7)

    geo_loss = dq_loss_fn(y_hat_flat, y_flat)

    pred_dqs = []
    for i in range(window_size - 1):
        pred_dq = y_hat_flat[i : i + 1]
        pred_dqs.append(pred_dq)

    tri_loss_val = torch.tensor(0.0, device=device)

    convert_to_pixel_coords = None
    pt1_squeezed = None
    pt2_squeezed = None
    pts_list = None
    n_valid_val = None

    use_any_tri_loss = args.get("use_triangulation_loss", False) or args.get(
        "use_residual_triangulation_loss", False
    )

    if use_any_tri_loss and pt1.numel() > 0 and pt2.numel() > 0:

        if n_valid is not None:
            if isinstance(n_valid, torch.Tensor):
                if n_valid.numel() == 1:
                    n_valid_val = n_valid.item()
                elif n_valid.numel() > 0:
                    n_valid_val = (
                        n_valid[0].item() if n_valid.dim() > 0 else n_valid.item()
                    )

    use_tri_loss = args.get("use_triangulation_loss", False)
    use_res_tri_loss = args.get("use_residual_triangulation_loss", False)
    use_any_tri_loss = use_tri_loss or use_res_tri_loss

    if use_any_tri_loss and pt1.numel() > 0 and pt2.numel() > 0:

        if n_valid is not None:
            if isinstance(n_valid, torch.Tensor):
                if n_valid.numel() == 1:
                    n_valid_val = n_valid.item()
                elif n_valid.numel() > 0:
                    n_valid_val = (
                        n_valid[0].item() if n_valid.dim() > 0 else n_valid.item()
                    )

        def convert_to_pixel_coords_fn(pt, cam_params, device):
            if pt.dim() == 2 and pt.shape[1] == 2:
                fx = cam_params.get("fx", 718.856)
                fy = cam_params.get("fy", 718.856)
                cx = cam_params.get("cx", 607.1928)
                cy = cam_params.get("cy", 185.2157)
                x = pt[:, 0] * fx + cx
                y = pt[:, 1] * fy + cy
                return torch.stack([x, y], dim=1)
            return pt

        if pt1.dim() == 3 and pt1.shape[0] > 1:
            pt1_squeezed = pt1[:, :n_valid_val, :] if n_valid_val else pt1
            pt2_squeezed = pt2[:, :n_valid_val, :] if n_valid_val else pt2
        else:
            pt1_squeezed = pt1[:n_valid_val, :] if n_valid_val else pt1
            pt2_squeezed = pt2[:n_valid_val, :] if n_valid_val else pt2

        if pt1_squeezed.shape[0] >= 4:
            cam_params = {
                "fx": 718.856,
                "fy": 718.856,
                "cx": 607.1928,
                "cy": 185.2157,
            }
            pt1_pixel = convert_to_pixel_coords_fn(pt1_squeezed, cam_params, device)
            pt2_pixel = convert_to_pixel_coords_fn(pt2_squeezed, cam_params, device)
            pts_list = [pt1_pixel, pt2_pixel]
        else:
            pts_list = None

        if use_tri_loss:
            if n_valid_val is not None and n_valid_val < 4:
                tri_loss_val = torch.tensor(0.0, device=device)
            elif (
                pts_list is not None and len(pts_list) >= 2 and pts_list[0].shape[0] > 0
            ):
                try:
                    if pt1.dim() == 3 and pt1.shape[0] > 1:
                        batch_size = pts_list[0].shape[0]
                        tri_losses = []
                        for b in range(batch_size):
                            pts_b = [pts[b] for pts in pts_list]
                            dq_b = [dq[b : b + 1] for dq in pred_dqs]

                            if pts_b[0].shape[0] < 4:
                                continue

                            try:
                                tri_loss_b = tri_loss_fn(
                                    pts_b,
                                    dq_b,
                                    K,
                                    normalize_coords=False,
                                    use_median=False,
                                )
                                if isinstance(tri_loss_b, torch.Tensor):
                                    if tri_loss_b.numel() > 1:
                                        tri_loss_b = tri_loss_b.mean()
                                    elif tri_loss_b.numel() == 1:
                                        tri_loss_b = tri_loss_b.flatten()[0]
                                tri_losses.append(tri_loss_b)
                            except Exception as e:
                                continue

                        if tri_losses:
                            tri_loss_val = sum(tri_losses) / len(tri_losses)
                        else:
                            tri_loss_val = torch.tensor(0.0, device=device)
                    else:
                        tri_loss_val = tri_loss_fn(
                            pts_list,
                            pred_dqs,
                            K,
                            normalize_coords=False,
                            use_median=False,
                        )
                        if isinstance(tri_loss_val, torch.Tensor):
                            if tri_loss_val.numel() > 1:
                                tri_loss_val = tri_loss_val.mean()
                            elif tri_loss_val.numel() == 1:
                                tri_loss_val = tri_loss_val.flatten()[0]
                except Exception as e:
                    tri_loss_val = torch.tensor(0.0, device=device)
            else:
                tri_loss_val = torch.tensor(0.0, device=device)

    residual_tri_loss_val = torch.tensor(0.0, device=device)
    if (
        use_res_tri_loss
        and pt1.numel() > 0
        and pt2.numel() > 0
        and n_valid_val is not None
        and n_valid_val >= 4
        and pts_list is not None
    ):
        try:
            if pt1.dim() == 3 and pt1.shape[0] > 1:
                batch_size = pts_list[0].shape[0]
                window_poses = window_size - 1
                residual_losses = []
                for b in range(batch_size):
                    pts_b = [pts[b] for pts in pts_list]

                    pred_dq_b = y_hat_flat[b * window_poses : b * window_poses + 1]
                    gt_dq_b = y_flat[b * window_poses : b * window_poses + 1]

                    if pts_b[0].shape[0] < 4:
                        continue
                    try:
                        res_loss_b = residual_tri_loss_fn(
                            pts_b,
                            [pred_dq_b],
                            [gt_dq_b],
                            K,
                            normalize_coords=False,
                        )
                        if isinstance(res_loss_b, torch.Tensor):
                            if res_loss_b.numel() > 1:
                                res_loss_b = res_loss_b.mean()
                            elif res_loss_b.numel() == 1:
                                res_loss_b = res_loss_b.flatten()[0]
                        residual_losses.append(res_loss_b)
                    except Exception:
                        continue
                if residual_losses:
                    residual_tri_loss_val = sum(residual_losses) / len(residual_losses)
            else:
                pred_dqs_for_loss = []
                for i in range(window_size - 1):
                    pred_dqs_for_loss.append(y_hat_flat[i : i + 1])
                gt_dqs_for_loss = []
                for i in range(window_size - 1):
                    gt_dqs_for_loss.append(y_flat[i : i + 1])
                try:
                    residual_tri_loss_val = residual_tri_loss_fn(
                        pts_list,
                        pred_dqs_for_loss,
                        gt_dqs_for_loss,
                        K,
                        normalize_coords=False,
                    )
                    if isinstance(residual_tri_loss_val, torch.Tensor):
                        if residual_tri_loss_val.numel() > 1:
                            residual_tri_loss_val = residual_tri_loss_val.mean()
                        elif residual_tri_loss_val.numel() == 1:
                            residual_tri_loss_val = residual_tri_loss_val.flatten()[0]
                except Exception:
                    residual_tri_loss_val = torch.tensor(0.0, device=device)
        except Exception:
            residual_tri_loss_val = torch.tensor(0.0, device=device)
            import traceback

            traceback.print_exc()
            residual_tri_loss_val = torch.tensor(0.0, device=device)

    lambda_geo = args.get("lambda_geo", 1.0)
    lambda_tri = args.get("lambda_tri", 0.5)
    lambda_res_tri = args.get("lambda_res_tri", 0.5)

    # Ensure tri_loss_val is a tensor for proper computation
    if not isinstance(tri_loss_val, torch.Tensor):
        tri_loss_val = torch.tensor(tri_loss_val, device=device)
    if not isinstance(residual_tri_loss_val, torch.Tensor):
        residual_tri_loss_val = torch.tensor(residual_tri_loss_val, device=device)

    total_loss = (
        lambda_geo * geo_loss
        + lambda_tri * tri_loss_val
        + lambda_res_tri * residual_tri_loss_val
    )
    # total_loss = geo_loss

    def safe_item(tensor_or_float):
        if isinstance(tensor_or_float, torch.Tensor):
            if tensor_or_float.numel() > 1:
                return float(tensor_or_float.mean())
            return (
                float(tensor_or_float.flatten()[0])
                if tensor_or_float.numel() == 1
                else float(tensor_or_float)
            )
        return float(tensor_or_float)

    loss_dict = {
        "total": safe_item(total_loss),
        "geo": safe_item(geo_loss),
        "tri": safe_item(tri_loss_val),
        "res_tri": safe_item(residual_tri_loss_val),
    }

    return total_loss, loss_dict


def get_camera_intrinsics(cam_params):
    fx = cam_params["fx"]
    fy = cam_params["fy"]
    cx = cam_params["cx"]
    cy = cam_params["cy"]

    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    return K


if __name__ == "__main__":
    args = {
        "data_dir": "data",
        "bsize": 8,
        "val_split": 0.1,
        "window_size": 3,
        "overlap": 2,
        "optimizer": "Adam",
        # "lr": 5e-7,
        "lr": 1e-6,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "epoch": 300,
        "weighted_loss": None,
        "pretrained_ViT": False,
        "checkpoint_path": "checkpoints/Exp53_tri",
        "checkpoint": "base.pth",
        "use_dual_quaternion": True,
        "use_triangulation_loss": False,
        "use_residual_triangulation_loss": True,
        "lambda_geo": 1.0,
        "lambda_tri": 0.01,
        "lambda_res_tri": 0.5,
    }

    model_params = {
        "dim": 384,
        "image_size": (224, 672),
        "patch_size": 16,
        "attention_type": "divided_space_time",
        "num_frames": args["window_size"],
        "num_classes": 6 * (args["window_size"] - 1),
        "depth": 16,
        "heads": 6,
        "dim_head": 64,
        "attn_dropout": 0.2,
        "ff_dropout": 0.2,
        "time_only": False,
    }
    args["model_params"] = model_params

    if not os.path.exists(args["checkpoint_path"]):
        os.makedirs(args["checkpoint_path"])

    with open(os.path.join(args["checkpoint_path"], "args.pkl"), "wb") as f:
        pickle.dump(args, f)
    with open(os.path.join(args["checkpoint_path"], "args.txt"), "w") as f:
        f.write(json.dumps(args))

    TensorBoardWriter = SummaryWriter(log_dir=args["checkpoint_path"])

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

    print("Using CUDA: ", torch.cuda.is_available())
    print("Loading data...")
    dataset = KITTI(
        window_size=args["window_size"],
        overlap=args["overlap"],
        transform=preprocess,
        train=True,
    )

    cam_K = get_camera_intrinsics(dataset.cam_params)
    if torch.cuda.is_available():
        cam_K = cam_K.cuda()

    nb_val = round(args["val_split"] * len(dataset))

    train_data, val_data = random_split(dataset, [len(dataset) - nb_val, nb_val])

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args["bsize"],
        num_workers=4,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=4,
    )

    print("Building model...")
    model, args = build_model(args, model_params)

    n = sum([param.nelement() for param in model.parameters()])
    print(f"Model parameters: {n}")

    criterion = torch.nn.MSELoss()
    tri_loss_fn = TriangulationLoss()
    residual_tri_loss_fn = (
        ResidualTriangulationLoss()
        if args.get("use_residual_triangulation_loss", False)
        else None
    )

    optimizer = get_optimizer(model.parameters(), args)

    print(20 * "--" + " Training with Triangulation Loss " + 20 * "--")
    train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        TensorBoardWriter,
        args,
        tri_loss_fn,
        cam_K,
        residual_tri_loss_fn,
    )
