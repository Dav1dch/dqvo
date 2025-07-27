import os
import pickle
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from tqdm import tqdm

from build_model import build_model
from datasets.kitti import KITTI

# from timesformer.models.vit import CrossViT, CrossViT_loop, VisionTransformer
from timesformer.models.vit_seq import CrossViT
from timesformer.models.mamba import CrossVisionMamba

checkpoint_path = "checkpoints/Exp30"
checkpoint_name = "checkpoint_best"
# sequences = ['00',"01", '02',"03", "04", "05", "06", "07", '08', '09', "10"]
sequences = ["01", "03", "04", "05", "06", "07", "10"]
# sequences = ["07"]
# sequences = ["03", "07"]

device = "cuda" if torch.cuda.is_available() else "cpu"

# read hyperparameters and configuration
with open(os.path.join(checkpoint_path, "args.pkl"), "rb") as f:
    args = pickle.load(f)
f.close()
model_params = args["model_params"]
args["checkpoint_path"] = checkpoint_path
print(args)

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

# build and load model
# model = VisionTransformer(img_size=model_params["image_size"],
#                           num_classes=model_params["num_classes"],
#                           patch_size=model_params["patch_size"],
#                           embed_dim=model_params["dim"],
#                           depth=model_params["depth"],
#                           num_heads=model_params["heads"],
#                           mlp_ratio=4,
#                           qkv_bias=True,
#                           norm_layer=partial(nn.LayerNorm, eps=1e-6),
#                           drop_rate=0.,
#                           attn_drop_rate=0.,
#                           drop_path_rate=0.1,
#                           num_frames=model_params["num_frames"],
#                           attention_type=model_params["attention_type"])


# model = CrossViT(
#     image_height=model_params["image_size"][0],
#     image_width=model_params["image_size"][1],
#     patch_size=model_params["patch_size"],
#     num_classes=256,
#     dim=model_params["dim"],
#     depth=model_params["depth"],
#     heads=model_params["heads"],
#     mlp_dim=1024,
# )

model = CrossVisionMamba(
    image_height=model_params["image_size"][0],
    image_width=model_params["image_size"][1],
    patch_size=model_params["patch_size"],
    num_classes=1000,
    # patch_size=16,
    embed_dim=192,
    depth=model_params["depth"],
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    final_pool_type="mean",
    if_abs_pos_embed=True,
    if_rope=False,
    if_rope_residual=False,
    bimamba_type="V2",
    if_cls_token=False,
    use_double_cls_token=False,
)


checkpoint = torch.load(
    os.path.join(args["checkpoint_path"], "{}.pth".format(checkpoint_name)),
    map_location=torch.device(device),
    weights_only=True,
)
print(args["checkpoint_path"])
model.load_state_dict(checkpoint["model_state_dict"])
if torch.cuda.is_available():
    model.cuda()

args["window_size"] = 3
args["overlap"] = 2

for sequence in sequences:
    # test dataloader
    dataset = KITTI(
        transform=preprocess,
        sequences=[sequence],
        # window_size=2,
        # overlap=1,
        window_size=args["window_size"],
        overlap=args["overlap"],
    )
    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=10,
        shuffle=False,
    )

    with tqdm(test_loader, unit="batch") as batchs:
        pred_poses = np.zeros((1, args["window_size"] - 1, 6))
        batchs.set_description(f"Sequence {sequence}")
        for images, pt1, pt2, gt in batchs:
            if torch.cuda.is_available():
                images, gt = images.cuda(), gt.cuda()

                with torch.no_grad():
                    model.eval()
                    model.training = False
                    src = list(range(images.shape[0]))
                    dst = list(range(1, images.shape[0] + 1))
                    # g = dgl.graph((src, dst)).to("cuda")

                    # predict pose
                    pred_pose = model(images.float()).cpu().detach().numpy()
                    pred_pose = np.reshape(
                        pred_pose, (images.shape[0], args["window_size"] - 1, 6)
                    )
                    # pred_pose = np.expand_dims(pred_pose,axis=0)
                    pred_poses = np.concatenate((pred_poses, pred_pose), axis=0)

    # save as numpy array
    pred_poses = pred_poses[1:, :, :]

    save_dir = os.path.join(args["checkpoint_path"], checkpoint_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save(os.path.join(save_dir, "pred_poses_{}.npy".format(sequence)), pred_poses)
