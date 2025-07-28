import math
import argparse
from pathlib import Path
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast

# -----------------------------------------------------------------------------
# 1. MaskIntegration 
# -----------------------------------------------------------------------------

class MaskIntegration(nn.Module):
    def __init__(self):
        super().__init__()
        # 这个模块将mask用于淡化背景区域的效果

    def forward(self, x, mask):
        # mask: [B, 1, H, W], x: [B, C, H, W]
        # 将mask调整为与x相同的大小
        mask_resized = F.interpolate(mask, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 通过mask来调节图像权重，背景区域逐渐变为较低的权重
        # 我们可以将mask的值映射到一个较低的范围，比如[0, 1] -> [0.2, 1]
        mask_weighted = mask_resized * 0.8 + 0.2  # 让背景区域的值更小
        # 将图像和调整后的mask结合
        x_out = x * mask_weighted
        return x_out


# -----------------------------------------------------------------------------
# 2.  Layers & Blocks (修改部分通道数以适应额外mask通道)
# -----------------------------------------------------------------------------

def _make_divisible(v, divisor=8):
    return int(math.ceil(v / divisor) * divisor)

class Conv2d_BN(nn.Sequential):
    def __init__(self, inp, oup, k=1, s=1, p=0, g=1, act=False):
        super().__init__(
            nn.Conv2d(inp, oup, k, s, p, groups=g, bias=False),
            nn.BatchNorm2d(oup)
        )
        if act:
            self.add_module('act', nn.GELU())

class GhostModule(nn.Module):
    """Ghost Convolution (Ma et al.)"""
    def __init__(self, inp, oup, k=1, s=1, ratio=2, dw_size=3, act=True):
        super().__init__()
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)
        self.primary = nn.Sequential(
            nn.Conv2d(inp, init_channels, k, s, k//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.GELU() if act else nn.Identity()
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2,
                      groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.GELU() if act else nn.Identity()
        )
        self.out_channels = oup

    def forward(self, x):
        y = self.primary(x)
        y = torch.cat([y, self.cheap(y)], dim=1)
        return y[:, :self.out_channels]

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),  # 激活函数
            nn.Linear(in_channels // reduction, in_channels, bias=False),  # 恢复维度
            nn.Sigmoid()  # 激活函数
        )

    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入的尺寸
        y = self.avg_pool(x).view(b, c)  # 全局平均池化
        y = self.fc(y).view(b, c, 1, 1)  # 经过全连接层
        return x * y.expand_as(x)  # 按照通道权重调整输入


class ResidualAdd(nn.Module):
    """带可选投影的残差加法"""
    def __init__(self, fn, in_c, out_c, stride):
        super().__init__()
        self.fn = fn
        if in_c != out_c or stride != 1:
            self.proj = nn.Sequential(
                nn.AvgPool2d(stride) if stride != 1 else nn.Identity(),
                Conv2d_BN(in_c, out_c, k=1)
            )
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(x) + self.fn(x)


class LiteBlock(nn.Module):
    def __init__(self, inp, oup, stride=1, se=True):
        super().__init__()
        hidden = _make_divisible(inp * 2, 8)

        self.token_mixer = nn.Sequential(
            GhostModule(inp, inp, k=3, s=stride, ratio=2, act=True),
            #ECA(inp) if eca else nn.Identity()
            SEBlock(inp) if se else nn.Identity()
        )
        channel_mixer = nn.Sequential(
            GhostModule(inp, hidden, act=True),
            GhostModule(hidden, oup, act=False)
        )
        self.residual = ResidualAdd(channel_mixer, inp, oup, stride=1)

    def forward(self, x):
        x = self.token_mixer(x)
        return self.residual(x)

class LiteGabor(nn.Module):
    def __init__(self, lr_mult=0.1):
        super().__init__()
        # 生成 3 个 3×3 的经典 Gabor 核
        kernels = []
        for theta in [0, math.pi/4, math.pi/2]:
            kernel = self._gabor_kernel(theta)
            kernels.append(kernel)
        k = torch.stack(kernels)  # (3,3,3)
        k = k.unsqueeze(1)  # (3,1,3,3)
        self.weight = nn.Parameter(k)  # (3,1,3,3)
        self.lr_mult = lr_mult
        
        # Learnable gating parameters
        self.gate_weight = nn.Parameter(torch.ones(1))  # A single scalar gate for all Gabor features

    def _gabor_kernel(self, theta, sigma=2.0, lam=3.0, gamma=0.5):
        y, x = torch.meshgrid(torch.arange(-1, 2, dtype=torch.float32),
                             torch.arange(-1, 2, dtype=torch.float32),
                             indexing='ij')
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        g = torch.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * xr / lam)
        return g

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 若输入通道不足 3，则复制；超出 3 可只取前三或做分组卷积
        if c < 3:
            x = x.repeat(1, (3 + c - 1) // c, 1, 1)[:, :3]

        # Gabor convolution
        gabor_features = F.conv2d(x, self.weight * self.lr_mult, padding=1, groups=3)
        
        # Noise level computation (e.g., variance)
        noise_level = torch.var(gabor_features, dim=(2, 3), keepdim=True)  # Variance over spatial dimensions

        # The gate adjusts the weight dynamically
        gate = torch.sigmoid(self.gate_weight)  # Sigmoid ensures the gate is between 0 and 1
        
        # Use the noise level to modulate the gate (higher variance implies lower gate)
        dynamic_gate = gate * (1 - noise_level)  # Reduce the gate value in noisy areas

        # Apply the dynamic gate to the Gabor features
        gated_gabor_features = gabor_features * dynamic_gate

        return gated_gabor_features


class LiteGabor2(nn.Module):
    def __init__(self, lr_mult=0.1, kernel_size=3, num_kernels=3):
        super().__init__()
        assert kernel_size in [3, 5, 7], "kernel_size must be 3, 5 or 7"
        assert num_kernels in [3, 4, 6, 8], "num_kernels must be 3, 4, 6 or 8"
        
        self.kernel_size = kernel_size
        self.num_kernels = num_kernels
        self.padding = kernel_size // 2  # 保持输出尺寸不变
        
        # 生成不同方向的Gabor核（均匀分布在0到π之间）
        kernels = []
        for i in range(num_kernels):
            theta = math.pi * i / num_kernels  # 均匀分布的角度
            kernel = self._gabor_kernel(theta, kernel_size=kernel_size)
            kernels.append(kernel)
        
        k = torch.stack(kernels)  # (num_kernels, kernel_size, kernel_size)
        k = k.unsqueeze(1)  # (num_kernels, 1, kernel_size, kernel_size)
        self.weight = nn.Parameter(k)
        self.lr_mult = lr_mult
        
        # 可学习的门控参数（每个核一个门控）
        self.gate_weight = nn.Parameter(torch.ones(num_kernels, 1, 1))

    def _gabor_kernel(self, theta, sigma=2.0, lam=3.0, gamma=0.5, kernel_size=3):
        # 创建网格坐标
        r = (kernel_size - 1) // 2
        y, x = torch.meshgrid(torch.arange(-r, r+1, dtype=torch.float32),
                             torch.arange(-r, r+1, dtype=torch.float32),
                             indexing='ij')
        
        # Gabor函数计算
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        g = torch.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * xr / lam)
        return g

    def forward(self, x):
        b, c, h, w = x.shape
        
        # 若输入通道不足num_kernels，则复制；超出则只取前num_kernels
        if c < self.num_kernels:
            x = x.repeat(1, (self.num_kernels + c - 1) // c, 1, 1)[:, :self.num_kernels]
        elif c > self.num_kernels:
            x = x[:, :self.num_kernels]

        # Gabor卷积（分组卷积，每组一个核）
        gabor_features = F.conv2d(x, self.weight * self.lr_mult, 
                                 padding=self.padding, groups=self.num_kernels)
        
        # 噪声水平计算（每个核的空间方差）
        noise_level = torch.var(gabor_features, dim=(2, 3), keepdim=True)  # [B, num_kernels, 1, 1]
        
        # 动态门控（每个核独立门控）
        gate = torch.sigmoid(self.gate_weight)  # [num_kernels, 1, 1]
        dynamic_gate = gate * (1 - noise_level)  # [B, num_kernels, 1, 1]
        
        # 应用门控
        gated_gabor_features = gabor_features * dynamic_gate

        return gated_gabor_features

class GaborAttention(nn.Module):
    """
    通过Gabor滤波器组生成空间注意力图来增强特征。
    """
    def __init__(self, in_channels, kernel_size=7, num_kernels=8):
        super().__init__()
        self.num_kernels = num_kernels
        self.padding = kernel_size // 2

        # 创建固定的Gabor核作为特征提取器
        kernels = []
        for i in range(num_kernels):
            theta = math.pi * i / num_kernels
            kernel = self._gabor_kernel(theta, kernel_size=kernel_size)
            kernels.append(kernel)
        k = torch.stack(kernels).unsqueeze(1)
        self.register_buffer('gabor_kernels', k)

        # 用于从Gabor响应生成注意力图的小型网络
        self.attention_net = nn.Sequential(
            nn.BatchNorm2d(num_kernels),
            nn.Conv2d(num_kernels, num_kernels, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_kernels, 1, 1, bias=False),
            nn.Sigmoid()
        )

    def _gabor_kernel(self, theta, sigma=2.0, lam=3.0, gamma=0.5, kernel_size=3):
        r = (kernel_size - 1) // 2
        y, x = torch.meshgrid(torch.arange(-r, r+1, dtype=torch.float32),
                             torch.arange(-r, r+1, dtype=torch.float32),
                             indexing='ij')
        
        xr = x * math.cos(theta) + y * math.sin(theta)
        yr = -x * math.sin(theta) + y * math.cos(theta)
        g = torch.exp(-(xr**2 + gamma**2 * yr**2) / (2 * sigma**2)) * torch.cos(2 * math.pi * xr / lam)
        return g

    def forward(self, x):
        # 将输入特征图转为灰度图，用于Gabor滤波
        x_grey = torch.mean(x, dim=1, keepdim=True)
        # 应用Gabor滤波器
        gabor_responses = F.conv2d(x_grey, self.gabor_kernels, padding=self.padding)
        # 生成空间注意力图
        attention_map = self.attention_net(gabor_responses)
        # 将注意力图应用到原始特征图上
        return x * attention_map

class GaborBlock(nn.Module):
    """
    一个更强大的Gabor模块，采用残差瓶颈结构，并嵌入GaborAttention。
    """
    def __init__(self, in_channels, expand_ratio=2, kernel_size=7, num_kernels=8):
        super().__init__()
        hidden_channels = _make_divisible(int(in_channels * expand_ratio), 8)
        
        self.gabor_path = nn.Sequential(
            Conv2d_BN(in_channels, hidden_channels, act=True),
            GaborAttention(hidden_channels, kernel_size, num_kernels),
            Conv2d_BN(hidden_channels, in_channels, act=False)
        )

    def forward(self, x):
        return x + self.gabor_path(x)

class RepViTTinyWithMask(nn.Module):
    def __init__(self, num_classes=5115, alpha=0.35):
        super().__init__()
        cfgs = [(64,2),(128,2),(160,2),(256,1),(384,1)]
        in_c = _make_divisible(32*alpha, 8)
 
        self.mask_integration = MaskIntegration()
        self.conv1 = Conv2d_BN(3, 3, k=3, s=2, p=1, act=True) 
        #self.gabor = LiteGabor2(kernel_size=7, num_kernels=8)
        self.gabor = GaborBlock(in_channels=3, kernel_size=7, num_kernels=4)
        self.conv2 = Conv2d_BN(3, in_c, k=1, act=True)

        blocks = []
        for c, s in cfgs:
            out_c = _make_divisible(c*alpha, 8) 
            blocks.append(LiteBlock(in_c, out_c, stride=s))
            in_c = out_c
        self.blocks = nn.ModuleList(blocks)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_c, num_classes, bias=False)

    def forward(self, x):

        #x = self.mask_integration(x, mask)
        x = self.conv1(x)
        x = self.gabor(x)
        x = self.conv2(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.pool(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out


def repvit_tiny_0_35(num_classes=5115):
    return RepViTTinyWithMask(num_classes=num_classes)


# from thop import profile
# import torch

# model = repvit_tiny_0_35().cpu()
# model.eval()
# #print(model)
# x = torch.randn(1, 3, 320, 320)
# #mask = torch.ones(1, 1, 320, 320)

# #flops, params = profile(model, inputs=(x, mask))
# flops, params = profile(model, inputs=(x,))
# print(f"FLOPs: {flops / 1e9:.2f} GFLOPs")
# print(f"Params: {params / 1e6:.2f} M")



# from torchviz import make_dot
# import torch
# import os
# import graphviz

# # 设置 Graphviz 的路径（替换成你的实际路径）
# os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

# model = repvit_tiny_0_35()
# # 创建虚拟输入
# x = torch.randn(1, 3, 224, 224)
# mask = torch.randn(1, 1, 224, 224)
# # 生成计算图
# y = model(x, mask)
# dot = make_dot(y, params=dict(model.named_parameters()))
# dot.render('repvit_tiny_with_mask', format='png')