# coding: utf-8
import os
from PIL import Image
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from timesformer.models.mamba_gated import CrossVisionMamba

import torchvision.transforms as transforms
import pickle

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Device: {}".format(device))

checkpoint_path = "checkpoints/Exp51"
checkpoint_name = "checkpoint_best_copy"

# 加载模型配置
with open(os.path.join(checkpoint_path, "args.pkl"), "rb") as f:
    args = pickle.load(f)
model_params = args["model_params"]

# 预处理操作
preprocess = transforms.Compose([
    transforms.Resize((model_params["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.34721234, 0.36705238, 0.36066107],
        std=[0.30737526, 0.31515116, 0.32020183],
    ),
])

# 创建CrossVisionMamba模型
model = CrossVisionMamba(
    image_height=model_params["image_size"][0],
    image_width=model_params["image_size"][1],
    patch_size=model_params["patch_size"],
    num_classes=1000,
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
).to(device).eval()
# 加载预训练权重
checkpoint_file = os.path.join(checkpoint_path, f"{checkpoint_name}.pth")
if os.path.exists(checkpoint_file):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_file}")
else:
    print(f"Warning: Checkpoint file {checkpoint_file} not found, using random weights")

print("Model created successfully")

# 简化版本：直接使用pytorch_grad_cam
from pytorch_grad_cam import GradCAM, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

class SimpleCrossVisionMambaCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        
        # 注册前向和反向钩子
        self.forward_hook = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_hook = self.target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        # CrossVisionMamba 层的输出是元组 (hidden_states, residual)
        # 我们只需要 hidden_states 部分
        if isinstance(output, tuple):
            self.activations = output[0].detach()  # 获取 hidden_states
        else:
            self.activations = output.detach()
        print(f"Activations shape: {self.activations.shape}")
    
    def _backward_hook(self, module, grad_input, grad_output):
        # 处理可能的元组梯度
        if grad_output is not None:
            if isinstance(grad_output, tuple):
                # 取第一个有效的梯度张量
                for grad in grad_output:
                    if grad is not None:
                        self.gradients = grad.detach()
                        break
            else:
                self.gradients = grad_output.detach()
            if self.gradients is not None:
                print(f"Gradients shape: {self.gradients.shape}")
    def generate_cam(self, input_tensor, target_index=0):
        """生成CAM热图"""
        # 清除之前的激活和梯度
        self.activations = None
        self.gradients = None
        
        # 前向传播
        output = self.model(input_tensor)
        print(f"Model output shape: {output.shape}")
        
        # 创建目标
        target = torch.zeros_like(output)
        target[0, 0, target_index] = 1.0  # 针对第一个时间步的特定维度
        
        # 计算损失
        loss = (output * target).sum()
        print(f"Loss: {loss.item()}")
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        # 检查是否成功获取激活和梯度
        if self.activations is None or self.gradients is None:
            print("Error: Failed to capture activations or gradients")
            return None
        
        print(f"Activations shape: {self.activations.shape}")
        print(f"Gradients shape: {self.gradients.shape}")
        
        # 根据实际的张量维度调整 Grad-CAM 计算
        if len(self.activations.shape) == 4:
            # 卷积层的输出: (batch_size, channels, height, width)
            # 在空间维度上平均梯度
            weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)  # 平均空间维度
            cam = torch.sum(weights * self.activations, dim=1)  # 在通道维度上加权求和
            
            # 确保热图是2D的
            if len(cam.shape) == 3:
                cam = cam.squeeze(0)  # 移除批次维度
            elif len(cam.shape) == 4:
                cam = cam.squeeze(0).squeeze(0)  # 移除批次和通道维度
            
        elif len(self.activations.shape) == 3:
            # 序列层的输出: (batch_size, sequence_length, features)
            # 这种层不适合生成图像热图
            print("Warning: Selected layer produces sequence output, not suitable for image heatmap")
            return None
            
        else:
            print(f"Unsupported activation shape: {self.activations.shape}")
            return None
        
        # ReLU和归一化
        cam = torch.relu(cam)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        print(f"CAM shape after processing: {cam.shape}")
        return cam.detach().cpu().numpy()

    def generate_spatial_cam(self, input_tensor, target_index=0):
        """为序列模型生成空间热图"""
        # 清除之前的激活和梯度
        self.activations = None
        self.gradients = None
        
        # 前向传播
        output = self.model(input_tensor)
        print(f"Model output shape: {output.shape}")
        
        # 创建目标
        target = torch.zeros_like(output)
        target[0, 0, target_index] = 1.0
        
        # 计算损失
        loss = (output * target).sum()
        print(f"Loss: {loss.item()}")
        
        # 反向传播
        self.model.zero_grad()
        loss.backward()
        
        if self.activations is None or self.gradients is None:
            print("Error: Failed to capture activations or gradients")
            return None
        
        print(f"Activations shape: {self.activations.shape}")
        print(f"Gradients shape: {self.gradients.shape}")
        
        # 如果激活是序列特征，尝试将其映射回图像空间
        if len(self.activations.shape) == 3:
            # (batch_size, sequence_length, features)
            # 将序列特征映射回图像网格
            B, seq_len, features = self.activations.shape
            
            # 获取图像patch的网格尺寸
            grid_h = model.patch_embed.grid_size[0]
            grid_w = model.patch_embed.grid_size[1]
            
            # 计算每个patch的重要性权重
            weights = torch.mean(self.gradients, dim=1, keepdim=True)  # (B, 1, features)
            patch_importance = torch.sum(weights * self.activations, dim=2)  # (B, seq_len)
            
            # 将序列重要性映射回图像网格
            heatmap = patch_importance.view(B, grid_h, grid_w)
            
            # 使用第一个批次
            cam = heatmap[0].detach().cpu().numpy()
            
        else:
            print(f"Unsupported activation shape: {self.activations.shape}")
            return None
        
        # ReLU和归一化
        cam = np.maximum(cam, 0)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


    
    def cleanup(self):
        """清理钩子"""
        self.forward_hook.remove()
        self.backward_hook.remove()


# 加载图像
# fnames = ['./data/sequences_jpg/01/image_2/000500.png', './data/sequences_jpg/01/image_2/000500.png', './data/sequences_jpg/01/image_2/000500.png']

# fnames = ['./data/sequences_jpg/00/image_2/000000.png', './data/sequences_jpg/00/image_2/000001.png', './data/sequences_jpg/00/image_2/000002.png']
fnames = ['./data/sequences_jpg/01/image_2/000498.png', './data/sequences_jpg/01/image_2/000499.png', './data/sequences_jpg/01/image_2/000500.png']
imgs = []

for fname in fnames:
    img = Image.open(fname).convert("RGB")
    img = preprocess(img)
    imgs.append(img)

# 创建输入张量 - 确保形状正确
# CrossVisionMamba期望: (batch_size, channels, sequence_length, height, width)
input_tensor = torch.stack(imgs).unsqueeze(0)  # 添加序列维度
input_tensor = input_tensor.to(device)
print(f"Input tensor shape: {input_tensor.shape}")

# 选择目标层
target_layer = model.patch_embed.proj
# target_layer = model.head[3]
print("Target layer selected")

# 创建CAM生成器
cam_generator = SimpleCrossVisionMambaCAM(model, target_layer)

try:
    # 生成热图
    print("Generating CAM...")
    heatmap = cam_generator.generate_cam(input_tensor)
    # heatmap = cam_generator.generate_spatial_cam(input_tensor)
    if heatmap is not None:
        print(f"Heatmap shape: {heatmap.shape}")
        
        # 读取原始图像
        original_img = cv2.imread(fnames[0])
        original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # 调整热图大小以匹配原始图像
        # heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        if len(heatmap.shape) == 2:
            # 2D热图
            heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
        elif len(heatmap.shape) == 3:
            # 3D热图，取第一个通道
            heatmap_resized = cv2.resize(heatmap[0], (original_img.shape[1], original_img.shape[0]))
        else:
            print(f"Unsupported heatmap shape: {heatmap.shape}")
        
        # 使用 matplotlib 的 colormap 创建 overlay
        def create_matplotlib_overlay(img, heatmap, colormap_name='winter'):
            """使用 matplotlib colormap 创建 overlay"""
            # 获取 colormap
            colormap = plt.get_cmap(colormap_name)
            
            # 应用 colormap
            heatmap_colored = colormap(heatmap)[:, :, :3]  # 移除 alpha 通道
            
            # 叠加到原始图像
            overlay = 0.5 * (img.astype(np.float32) / 255.0) + 0.5 * heatmap_colored
            overlay = np.clip(overlay, 0, 1)
            
            return (overlay * 255).astype(np.uint8)
        
        # 生成 overlay
        overlay = create_matplotlib_overlay(original_img, heatmap_resized, 'jet')
        
        # 只显示 overlay 图像
        plt.figure(figsize=(8, 6))
        plt.imshow(overlay)
        # plt.title("Grad-CAM Overlay (Ours)")
        plt.axis('off')
        
        # 保存图像
        plt.savefig("our.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Overlay with viridis colormap saved as crossvisionmamba_overlay_viridis.png")
    else:
        print("Failed to generate CAM")

       
    # if heatmap is not None:
    #     print(f"Heatmap shape: {heatmap.shape}")
        
    #     # 可视化结果
    #     plt.figure(figsize=(12, 4))
        
    #     # 显示原始图像
    #     plt.subplot(1, 3, 1)
    #     original_img = cv2.imread(fnames[0])
    #     original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    #     plt.imshow(original_img)
    #     plt.title("Original Image")
    #     plt.axis('off')
        
    #     # 显示热图
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(heatmap, cmap='jet')
    #     plt.title("Grad-CAM Heatmap")
    #     plt.axis('off')
        
    #     # 显示叠加结果
    #     plt.subplot(1, 3, 3)
    #     # 调整热图大小以匹配原始图像
    #     heatmap_resized = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    #     overlay = show_cam_on_image(
    #         original_img.astype(np.float32) / 255.0, 
    #         heatmap_resized
    #     )
    #     plt.imshow(overlay)
    #     plt.title("Overlay")
    #     plt.axis('off')
        
    #     plt.tight_layout()
    #     plt.savefig("crossvisionmamba_cam_result.png", dpi=300, bbox_inches='tight')
    #     plt.show()
        
    #     print("CAM generated successfully! Saved as crossvisionmamba_cam_result.png")
    # else:
    #     print("Failed to generate CAM")

except Exception as e:
    print(f"Error occurred: {e}")
    import traceback
    traceback.print_exc()

finally:
    # 清理钩子
    cam_generator.cleanup()
    print("Cleanup completed")
