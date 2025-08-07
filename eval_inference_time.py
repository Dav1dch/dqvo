import torch
import numpy as np
from torch.backends import cudnn

import tqdm
import torch.nn as nn

cudnn.benchmark = True

device = "cuda"

from timesformer.models.vit import VisionTransformer
from timesformer.models.vit_seq import CrossViT
from timesformer.models.mamba import CrossVisionMamba

from functools import partial

# model = ...

# build and load model


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

model_params = {
    "dim": 384,
    "image_size": (224, 672),  # (192, 640),
    # "image_size": (192, 640),  # (192, 640),
    "patch_size": 16,
    "attention_type": "divided_space_time",  # ['divided_space_time', 'space_only','joint_space_time', 'time_only']
    "num_frames": 2,
    "num_classes": 6 * (2 - 1),  # 6 DoF for each frame
    "depth": 16,
    "heads": 8,
    "dim_head": 64,
    "attn_dropout": 0.1,
    "ff_dropout": 0.1,
    "time_only": False,
}
# model = CrossVisionMamba(
#     image_height=model_params["image_size"][0],
#     image_width=model_params["image_size"][1],
#     patch_size=model_params["patch_size"],
#     num_classes=1000,
#     # patch_size=16,
#     embed_dim=192,
#     depth=model_params["depth"],
#     rms_norm=True,
#     residual_in_fp32=True,
#     fused_add_norm=True,
#     final_pool_type="mean",
#     if_abs_pos_embed=True,
#     if_rope=False,
#     if_rope_residual=False,
#     bimamba_type="V2",
#     if_cls_token=False,
#     use_double_cls_token=False,
# )

model = VisionTransformer(
    img_size=model_params["image_size"],
    num_classes=model_params["num_classes"],
    patch_size=model_params["patch_size"],
    embed_dim=model_params["dim"],
    depth=model_params["depth"],
    num_heads=model_params["heads"],
    mlp_ratio=4,
    qkv_bias=True,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    drop_rate=0.0,
    attn_drop_rate=0.0,
    drop_path_rate=0.1,
    num_frames=model_params["num_frames"],
    attention_type=model_params["attention_type"],
)
model = model.cuda().eval()

dummy_input = torch.randn(1, 3, 2, 224, 672).to(device)

starter = torch.cuda.Event(enable_timing=True)
ender = torch.cuda.Event(enable_timing=True)
# warm up

time = []
with torch.no_grad():
    for _ in range(10):
        _ = model(dummy_input)
    for _ in range(100):
        starter.record()
        _ = model(dummy_input)
        ender.record()
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        time.append(curr_time)

print(np.mean(time))
