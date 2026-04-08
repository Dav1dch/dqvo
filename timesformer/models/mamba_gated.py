# @Time : 2024/5/13
# @Author : Johnny Haytham
# @Location : Beijing
# @File : DeMamba.py
from einops import rearrange
import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple
from torch import Tensor
import torch.nn.functional as F
from typing import Optional
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba

from timesformer.models.spiral_form import generate_spiral_index_tensor

# from rope import *
from timm.models.layers import trunc_normal_, lecun_normal_
import random

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    print(ImportError)
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbedMix(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj_channel = nn.Conv2d(
            in_chans * 2, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.proj_add = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        B, C, T, H, W = x.shape
        x1 = x[:, :, 1, :, :]
        x2 = x[:, :, 0, :, :]
        x = rearrange(x, "b c t h w -> b (t c) h w")  # 4 2 3 192 640 -> 4 6 192 640
        x_concate = self.proj_channel(x)  # concate
        x_add = self.proj_add(x1 + x2)  # add
        result = self.norm_conv(x_concate + x_add)  #
        W = result.size(-1)
        result = result.flatten(2).transpose(1, 2)  # 8 480 384
        return result


class SafeSequencePatchAttentionWeights(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.5):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 确保头数整除特征维度
        if self.head_dim * num_heads != dim:
            # 调整头维度以确保整除
            self.head_dim = dim // num_heads
            self.adjusted_dim = self.head_dim * num_heads
            print(
                f"Warning: dim {dim} not divisible by num_heads {num_heads}. Using adjusted dim {self.adjusted_dim}"
            )
        else:
            self.adjusted_dim = dim

        # 线性变换层
        self.to_query = nn.Linear(dim, self.adjusted_dim)
        self.to_key = nn.Linear(dim, self.adjusted_dim)
        self.to_value = nn.Linear(dim, self.adjusted_dim)

        # 位置编码
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, dim))

        # 缩放因子
        self.scale = self.head_dim**-0.5

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.to_query.weight)
        nn.init.xavier_uniform_(self.to_key.weight)
        if self.to_query.bias is not None:
            nn.init.constant_(self.to_query.bias, 0)
        if self.to_key.bias is not None:
            nn.init.constant_(self.to_key.bias, 0)
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

    def forward(self, x, M):
        """
        安全的前向传播，避免索引越界

        Args:
            x: 输入张量，形状为 (B * M, L, dim)
            M: 每个图片的patch数量

        Returns:
            weights: 注意力权重，形状为 (B, L, M)
        """
        # 获取输入形状
        BM, L, dim = x.shape
        B = BM // M

        # 添加位置编码
        x = x + self.pos_encoding

        # 重塑输入: (B * M, L, dim) -> (B, M, L, dim)
        x_reshaped = x.view(B, M, L, dim)

        # 生成查询向量: 对每个样本的所有patch取平均，然后生成查询
        query_input = x_reshaped.mean(dim=1)  # (B, L, dim)
        query = self.to_query(query_input)  # (B, L, adjusted_dim)

        # 生成键: 保持原始形状
        key = self.to_key(x_reshaped)  # (B, M, L, adjusted_dim)
        # 生成值: 保持原始形状
        value = self.to_value(x_reshaped)  # (B, M, L, adjusted_dim)

        # 重塑查询: (B, L, adjusted_dim) -> (B, L, num_heads, head_dim) -> (B, num_heads, L, head_dim)
        query = query.view(B, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 重塑键: (B, M, L, adjusted_dim) -> (B, M, L, num_heads, head_dim) -> (B, num_heads, M, L, head_dim)
        key = key.view(B, M, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        # 计算注意力分数
        # (B, num_heads, L, head_dim) @ (B, num_heads, head_dim, M*L) -> (B, num_heads, L, M*L)
        key_reshaped = key.contiguous().view(B, self.num_heads, M * L, self.head_dim)
        attn_scores = torch.matmul(query, key_reshaped.transpose(-2, -1)) * self.scale

        # 重塑注意力分数: (B, num_heads, L, M*L) -> (B, num_heads, L, M, L)
        attn_scores = attn_scores.view(B, self.num_heads, L, M, L)

        # 对最后一个L维度取平均，得到每个patch的注意力分数
        attn_scores = attn_scores.mean(dim=-1)  # (B, num_heads, L, M)

        # 对M维度进行softmax，得到每个序列位置对每个patch的注意力权重
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, num_heads, L, M)

        # 应用dropout
        attn_weights = self.dropout(attn_weights)

        # 平均所有头的注意力权重，得到最终权重 (B, L, M)
        weights = attn_weights.mean(dim=1)  # (B, L, M)
        value = value.view(B, M, L, self.num_heads, self.head_dim).permute(
            0, 3, 1, 2, 4
        )
        # 重塑值: (B, num_heads, M, L, head_dim) -> (B, num_heads, M*L, head_dim)
        value_reshaped = value.contiguous().view(
            B, self.num_heads, M * L, self.head_dim
        )

        # 应用注意力权重到值上
        # (B, num_heads, L, M) -> (B, num_heads, L, M, 1)
        attn_weights_expanded = attn_weights.unsqueeze(-1)

        # 重塑值: (B, num_heads, M*L, head_dim) -> (B, num_heads, M, L, head_dim)
        value_reshaped = value_reshaped.view(B, self.num_heads, M, L, self.head_dim)

        # 应用注意力权重: (B, num_heads, L, M, 1) * (B, num_heads, M, L, head_dim) -> (B, num_heads, L, M, head_dim)
        weighted_value = attn_weights_expanded * value_reshaped.permute(0, 1, 3, 2, 4)

        # 合并所有头: (B, num_heads, L, M, head_dim) -> (B, L, M, adjusted_dim)
        weighted_value = (
            weighted_value.permute(0, 2, 3, 1, 4)
            .contiguous()
            .view(B, L, M, self.adjusted_dim)
        )

        return weighted_value


class PatchAttention(nn.Module):
    def __init__(self, dim, num_heads=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.query_proj = nn.Linear(dim, 1)  # 将特征映射到标量
        self.temperature = nn.Parameter(torch.ones(1))  # 可学习的温度参数

    def forward(self, x, M):
        BM, L, dim = x.shape
        B = BM // M

        # 重塑输入: (B * M, L, dim) -> (B, M, L, dim)
        x_reshaped = x.view(B, M, L, dim)

        # 计算注意力分数: 对每个位置应用线性变换
        attn_scores = self.query_proj(x_reshaped).squeeze(-1)  # (B, M, L)

        # 应用温度参数并计算softmax
        attn_scores = attn_scores / self.temperature
        weights = F.softmax(attn_scores, dim=1)  # 在patch维度上softmax

        # 转置得到 (B, L, M)
        weights = weights.permute(0, 2, 1)
        return weights


class PatchEmbedMamba(nn.Module):
    def __init__(
        self,
        # img_size=224,
        img_size=(224, 224),
        patch_size=16,
        stride=16,
        in_channels=3,
        embed_dim=256,
        norm_layer=None,
        flatten=True,
    ):
        super(PatchEmbedMamba, self).__init__()
        # img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)  # 将img_size和patch_size化成元组的形式
        self.img_size = img_size
        self.patch_size = patch_size
        # 一个patch形成一个grid（网格），这里记录网格的形状
        self.grid_size = (
            (img_size[0] - patch_size[0]) // stride + 1,
            (img_size[1] - patch_size[1]) // stride + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]  # 总共的patch个数
        self.flatten = flatten
        # 打patch的操作，实际为卷积的操作(为了不重复卷积，步长的大小理论上因该等于卷积核的大小）
        # self.proj = nn.Sequential(
        #     nn.InstanceNorm2d(3, affine=True),
        #     nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=stride),
        # )

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=stride
        )
        ssm_cfg = {}
        factory_kwargs = {"device": None, "dtype": None}

        self.layers = nn.ModuleList(
            [
                create_block(  # 对VisionMamba的Encoder进行初始化的操作
                    embed_dim,
                    ssm_cfg=None,
                    norm_epsilon=1e-5,
                    rms_norm=True,
                    residual_in_fp32=False,
                    fused_add_norm="mean",
                    layer_idx=i,
                    if_bimamba=False,
                    bimamba_type="V2",
                    drop_path=0.2,
                    if_divide_out=False,
                    init_layer_scale=None,
                    **factory_kwargs,
                )
                for i in range(3)
            ]
        )
        # self.rnn = nn.RNN(192, 192, 3, batch_first=True)
        # self.conv1d = nn.Conv1d(1, 1, embed_dim)
        self.weight_proj = SafeSequencePatchAttentionWeights(embed_dim, 8)
        self.norm = (
            norm_layer(embed_dim) if norm_layer else nn.Identity()
        )  # nn.Identity的输入等于输出，通常作为占位层使用

    def forward(self, x):
        B, C, L, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input img size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})"
        x = x.reshape(B * L, C, H, W)
        x = self.proj(x)  # B,C,H,W——>B,embed_dim,grid_size,grid_size
        if self.flatten:
            x = x.flatten(2).transpose(
                1, 2
            )  # B,embed_dim,grid_size,grid_size——>B,embed_dim,grid_size*grid_size——>B,grid_size*grid_size,embed_dim
        x = self.norm(x)
        x = x.reshape(B * self.grid_size[0] * self.grid_size[1], L, -1)
        # x, _ = self.rnn(x)
        residule = None
        for layer in self.layers:
            x, residule = layer(x, residule)
        x_weight = self.weight_proj(
            x[:, 1:].reshape(B * self.grid_size[0] * self.grid_size[1], (L - 1), -1),
            self.grid_size[0] * self.grid_size[1],
            # ).reshape(B * (L - 1), -1)
        ).reshape(B * (L - 1), self.grid_size[0] * self.grid_size[1], -1)
        x = x[:, 1:].reshape(B * (L - 1), self.grid_size[0] * self.grid_size[1], -1)
        return x, x_weight


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        super(Block, self).__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        self.mixer = mixer_cls(dim)  # 这其实是Mamba的部分固定参数的调用
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path)

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import Fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self,
        hidden_states: Tensor,  # 上一个时间状态的输出，也就是ht-1
        residual: Optional[Tensor] = None,
        inference_params=None,
    ):
        if not self.fused_add_norm:  # 如果fused_add_norm为False
            if residual is None:  # 如果残差为空，这个是if用于第一个block处理输入数据
                residual = hidden_states
            else:  # 如果残差不为空，这个if用于处理除了第一个block以外的所有block的操作
                residual = residual + self.drop_path(self.mixer(hidden_states))
                # 将residual的数据类型转化为self.norm.weight.dtype，将residual归一化后保存为hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:  # 如果指定self_residual的类型是float32的话
                residual = residual.to(torch.float32)

        else:  # 如果fused_add_norm不为False
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            )
            if residual is None:  # 如果残差为空，这个是if用于第一个block处理输入数据
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:  # 如果残差不为空，这个if用于处理除了第一个block以外的所有block的操作
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual


def create_block(
    d_model,  # token维度
    ssm_cfg=None,  # ssm模型的配置文件
    norm_epsilon=1e-5,  #
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,  # 是否使用双向mamba扫描
    bimamba_type="none",
    if_divide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:  # 如果使用双向mamba扫描
        bimamba_type = "v1"  # 这是一个模型的版本号
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(  # 代表着VIM Encoder对class token的拼接方式,cls token可以拼接到不同位置（所有token前面，所有token中间，...或是随机位置）
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_divide_out=if_divide_out,
        init_layer_scale=init_layer_scale,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(  # 对于class token的normalization函数
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs
    )  # eps用于避免归一化过程中分母为0的情况
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class CrossVisionMamba(nn.Module):
    def __init__(
        self,
        # img_size=224,
        image_height=244,
        image_width=244,
        patch_size=16,
        stride=16,
        depth=24,  # 需要构造的block的个数
        embed_dim=192,
        channels=3,
        num_classes=1000,  # 这里用imagenet做分类任务所以有1000个类，也就代表了最后的mlp的输出层包含1000个节点
        ssm_cfg=None,  # ssm的配置文件
        drop_rate=0.0,  # drop_rate是针对于dropout的频率（对某个节点进行失活的操作）
        drop_path_rate=0.2,  # drop_path_rate是针对drop_path的频率（对某个层进行失活的操作）
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,  # 是否使用rms_norm这种方法
        fused_add_norm=False,
        residual_in_fp32=False,  # 残差链接的时候是不是浮点型
        device=None,
        dtype=None,
        pt_hw_seq_len=14,  # 代表sequence的长度
        if_bidirectional=True,
        final_pool_type="none",  # 最后池化层的类型
        if_abs_pos_embed=False,  # 在位置编码的时候是不是需要用绝对值编码（有两种位置编码方式：1、直接给出的绝对值位置编码 2、可学习的位置编码）
        if_rope=False,  # rope也是一种对positionembeding的特殊编码方式
        if_rope_residual=False,  # 对 residual的rope 旨在增加鲁棒性
        flip_img_sequences_ratio=-1.0,  # image_squence的反转概率
        if_bimamba=False,
        bimamba_type="none",  # 表示使用的mamba的版本
        if_cls_token=False,  # 拼不拼clstoken
        if_divide_out=False,
        init_layer_scale=None,
        use_double_cls_token=False,
        use_middle_cls_token=False,
        **kwargs,
    ):  # 为了保证模型的可扩展性所以加一个**kwargs
        factory_kwargs = {"device": device, "dtype": dtype}
        # add factory_kwargs into kwargs
        kwargs.update(factory_kwargs)
        super(CrossVisionMamba, self).__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.if_bidirectional = if_bidirectional
        self.final_pool_type = final_pool_type
        self.if_abs_pos_embed = if_abs_pos_embed
        self.if_rope = if_rope
        self.if_rope_residual = if_rope_residual
        self.flip_img_sequences_ratio = flip_img_sequences_ratio
        self.if_cls_token = if_cls_token
        self.use_double_cls_token = (
            use_double_cls_token  # 这个拼接clstoken的方式是头拼一个尾拼一个
        )
        self.use_middle_cls_token = (
            use_middle_cls_token  # 这个拼接clstoken的方式是中间拼一个
        )
        self.num_tokens = 1 if if_cls_token else 0  # 表示拼了几个cls token进去？存疑

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = (
            embed_dim  # num_features for consistency with other models
        )
        self.spiral_indice = torch.tensor(
            generate_spiral_index_tensor(
                image_height // patch_size, image_width // patch_size
            )
        ).cuda()
        length = (image_height // patch_size) * (image_width // patch_size)
        # spiral_weight = (
        #     (torch.arange(1, length + 1) / (length * (length + 1) // 2))
        #     .unsqueeze(0)
        #     .cuda()
        # )
        # self.spiral_weight = nn.Parameter(spiral_weight)

        self.patch_embed = PatchEmbedMamba(
            img_size=(image_height, image_width),
            patch_size=patch_size,
            stride=stride,
            in_channels=channels,
            embed_dim=embed_dim,
            norm_layer=nn.LayerNorm,
        )
        num_patches = self.patch_embed.num_patches

        if if_cls_token:  # 如果使用cls token的话
            if use_double_cls_token:
                self.cls_token_head = nn.Parameter(
                    torch.zeros(1, 1, self.embed_dim)
                )  # 拼在token序列最前面的clstoken
                self.cls_token_tail = nn.Parameter(
                    torch.zeros(1, 1, self.embed_dim)
                )  # 拼在token序列最后面的clstoken
                self.num_tokens = 2  # 代表了拼了几个cls token
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
                # self.num_tokens = 1

        if if_abs_pos_embed:  # 如果使用给定的位置编码(给定绝对值)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + self.num_tokens, self.embed_dim)
            )
            self.pos_drop = nn.Dropout(p=drop_rate)

        # if if_rope:                                                  #Rope(Rotary Position Embedding)对于Position Embedding的翻转操作，（数据增强操作）
        #    half_head_dim = embed_dim // 2
        #    hw_seq_len = img_size // patch_size                      #高/宽方向的序列长度
        #    self.rope = VisionRotaryEmbeddingFast(
        #        dim=half_head_dim,
        #        pt_seq_len=pt_hw_seq_len,
        #        ft_seq_len=hw_seq_len
        #    )
        self.head = (
            nn.Sequential(
                nn.LayerNorm(self.num_features),
                nn.Linear(self.num_features, num_classes),
                nn.ReLU(),
                nn.Linear(self.num_classes, 6),
            )
            # nn.Linear(self.num_features, num_classes)
            if num_classes > 0
            else nn.Identity()
        )  # 这个是最终的分类头

        # drop path rate 随机失活一些东西，目的是让模型的鲁棒性更强，效果更好
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # 构建从start到end的等距张量，目的是为每层网络设置独立的drop_path_rate
        inter_dpr = [0.0] + dpr  # 第一层不需要dropout，所以要在最开始拼个0
        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

        self.layers = nn.ModuleList(
            [
                create_block(  # 对VisionMamba的Encoder进行初始化的操作
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_divide_out=if_divide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            embed_dim, eps=norm_epsilon, **factory_kwargs
        )
        # trunc_normal_函数是一个用于对张量进行截断正态分布初始化的函数。它通常用于初始化神经网络的权重或偏置。
        if if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

        if if_cls_token:
            if use_double_cls_token:
                trunc_normal_(self.cls_token_head, std=0.02)
                trunc_normal_(self.cls_token_tail, std=0.02)
            else:
                trunc_normal_(self.cls_token, std=0.02)

        # 定义前向特征传播的方法

    def forward_features(
        self,
        x,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        x, weight = self.patch_embed(x)
        B, M, _ = x.shape
        x = x[:, self.spiral_indice, :]

        if self.if_cls_token:
            if self.use_double_cls_token:  # 在序列前后拼double_cls_token
                cls_token_head = self.cls_token_head.expand(
                    B, -1, -1
                )  # expend 是共享内存的拓展 并不是创建新的张量
                cls_token_tail = self.cls_token_tail.expand(B, -1, -1)

                token_position = [0, M + 1]
                x = torch.cat((cls_token_head, x, cls_token_tail), dim=1)
                M = x.shape[1]

            else:
                if self.use_middle_cls_token:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = M // 2
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                elif if_random_cls_token_position:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = random.randint(0, M)
                    x = torch.cat(
                        (x[:, :token_position, :], cls_token, x[:, token_position:, :]),
                        dim=1,
                    )
                    print("token_position: ", token_position)
                else:
                    cls_token = self.cls_token.expand(B, -1, -1)
                    token_position = 0
                    x = torch.cat((cls_token, x), dim=1)
                M = x.shape[1]

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        if (
            if_random_token_rank
        ):  # 是否要把所有的token序列打乱，如果打乱了的话自然要更新存储clstoken的位置

            # 生成随机 shuffle索引
            shuffle_indices = torch.randperm(
                M
            )  # torch.randperm(M)是用于生成一个从0到M-1的随机排列的整数序列的函数。

            if isinstance(token_position, list):
                print(
                    "original value: ",
                    x[0, token_position[0], 0],
                    x[0, token_position[1], 0],
                )
            else:
                print("original value: ", x[0, token_position, 0])
            print("original token_position: ", token_position)

            # 执行shuffle
            x = x[:, shuffle_indices, :]

            if isinstance(token_position, list):
                new_token_position = [
                    torch.where(shuffle_indices == token_position[i])[0].item()
                    for i in range(len(token_position))
                ]
                token_position = new_token_position
            else:
                token_position = torch.where(shuffle_indices == token_position)[
                    0
                ].item()

            if isinstance(token_position, list):
                print(
                    "new value: ",
                    x[0, token_position[0], 0],
                    x[0, token_position[1], 0],
                )
            else:
                print("new value: ", x[0, token_position, 0])
            print("new token_position: ", token_position)

        if_flip_img_suquences = False
        if (
            self.flip_img_sequences_ratio > 0
            and (self.flip_img_sequences_ratio - random.random()) > 1e-5
        ):
            x = x.flip(
                [1]
            )  # 会创建一个与张量 x 的形状相同的新张量，其中第一个维度的元素被翻转。翻转是指将第一个维度中的元素按相反的顺序重新排列。
            if_flip_img_suquences = True

        # mamba的整体部分
        residual = None
        hidden_states = x
        if (
            not self.if_bidirectional
        ):  # 只使用单向扫描（所以单向扫描就既可以选择正向单向扫描进行rope，也可以选择反向单项扫描进行rope）
            for layer in self.layers:

                if (
                    if_flip_img_suquences and self.if_rope
                ):  # 反转序列并使用加强版的position Embedding
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                # rope about
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residuals:
                        residual = self.rope(residual)

                if (
                    if_flip_img_suquences and self.if_rope
                ):  # 这里并不是跟上上段代码重复，而是filp了之后要再反转过来
                    hidden_states = hidden_states.flip([1])
                    if residual is not None:
                        residual = residual.flip([1])

                hidden_states, residual = layer(
                    hidden_states,
                    residual,
                    inference_params=inference_params,
                )

        else:  # 如果采用双向扫描
            for i in range(len(self.layers) // 2):
                if self.if_rope:
                    hidden_states = self.rope(hidden_states)
                    if residual is not None and self.if_rope_residuals:
                        residual = self.rope(residual)

                hidden_states_f, residual_f = self.layers[i * 2](
                    hidden_states, residual, inference_params=inference_params
                )
                hidden_state_b, residual_b = self.layers[i * 2 + 1](
                    hidden_states.flip([1]),
                    None if residual is None else residual.flip([1]),
                    inference_params=inference_params,
                )
                hidden_states = hidden_states_f + hidden_state_b.flip([1])
                residual = residual_f + residual_b.flip([1])

        if not self.fused_add_norm:  # 如果不使用fused_add_norm
            if residual is None:  # 如果残差为空
                residual = hidden_states
            else:  # 如果残差不为空
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = (
                rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            )
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        # return only cls token if it exists
        if self.if_cls_token:
            if self.use_double_cls_token:
                return (
                    hidden_states[:, token_position[0], :]
                    + hidden_states[:, token_position[1], :]
                ) / 2
            else:
                if self.use_middle_cls_token:
                    return hidden_states[:, token_position, :]
                elif if_random_cls_token_position:
                    return hidden_states[:, token_position, :]
                else:
                    return hidden_states[:, token_position, :]

        if self.final_pool_type == "none":
            return hidden_states[:, -1, :]  # 这个切片是为了之后的mlp所做出的妥协
        elif self.final_pool_type == "mean":
            # weight = self.spiral_weight.flip(dims=[1])
            # weight = self.spiral_weight
            # hidden_states = hidden_states * weight.repeat(
            #     (hidden_states.shape[0], 1)
            # ).unsqueeze(-1)

            hidden_states = torch.mul(
                hidden_states,
                F.silu(
                    weight[:, self.spiral_indice]
                    # + self.spiral_weight
                    # weight
                ),
            )
            # return hidden_states.mean(dim=1)
            return hidden_states.sum(dim=1)
        elif self.final_pool_type == "max":
            return hidden_states
        elif self.final_pool_type == "all":
            return hidden_states
        else:
            raise NotImplementedError

    def forward(
        self,
        x,
        return_features=False,
        inference_params=None,
        if_random_cls_token_position=False,
        if_random_token_rank=False,
    ):
        # x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)
        b = x.size(0)
        l = x.size(2) - 1
        # x1 = x[:, :, :-1]
        # x2 = x[:, :, 1:]
        # x_new = torch.cat(
        #     (x1.reshape(b * l, 3, 1, 192, 640), x2.reshape(b * l, 3, 1, 192, 640)),
        #     dim=2,
        # )
        x = self.forward_features(
            x,
            inference_params,
            if_random_cls_token_position=if_random_cls_token_position,
            if_random_token_rank=if_random_token_rank,
        )
        if return_features:
            return x
        x = self.head(x).view(b, l, -1)
        if self.final_pool_type == "max":
            x = x.max(dim=1)[0]
        # return x.unsqueeze(1)
        return x


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CrossVisionMamba(
        patch_size=16,
        embed_dim=192,
        depth=24,
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        final_pool_type="mean",
        if_abs_pos_embed=True,
        if_rope=False,
        if_rope_residual=False,
        bimamba_type="V2",
        if_cls_token=True,
        use_double_cls_token=True,
    ).to(device)

    x = torch.randn(size=(4, 3, 7, 224, 224)).to(device)
    b = x.size(0)
    l = x.size(2) - 1
    x1 = x[:, :, :-1]
    x2 = x[:, :, 1:]
    x_new = torch.cat(
        (x1.reshape(b * l, 3, 1, 224, 224), x2.reshape(b * l, 3, 1, 224, 224)), dim=2
    )
    preds = model(x_new).view(b, l, -1)
    print(preds.shape)


if __name__ == "__main__":
    test()
