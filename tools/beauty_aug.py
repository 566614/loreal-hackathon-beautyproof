# -*- coding: utf-8 -*-
"""美妆域数据增强（domain-aware augmentation）。

为什么单独成模块
----------------
本赛题的真实世界痛点是：真实美妆图普遍带「滤镜 + 磨皮 + 重度 JPEG 重压缩 + 平台缩放」，
而我们的训练真实图只有 ~84 张（real_xhs 74 + real 10）。模型在真实美妆图上 66% 误报成 AI，
根因就是训练分布没覆盖「美颜后的真实照片」这种分布。

本模块的所有变换都只作用于「已合法取得的自产图」（即梦生成图 + 用户/队友提供的真实美妆图），
不引入任何第三方数据集，符合赛题「数据来源合规 + 原创性」要求。

手段：纯图像处理增强（机器学习标准做法），无外部数据、无预训练权重依赖。
"""
import io
import random

import numpy as np
from PIL import Image
import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224


class RandomJPEG:
    """模拟社交平台/微信的 JPEG 重压缩伪影（真实美妆图几乎都经过这步）。"""

    def __init__(self, q=(55, 92)):
        self.q = q

    def __call__(self, img):
        q = random.randint(int(self.q[0]), int(self.q[1]))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class RandomResizeNoise:
    """模拟「被缩小再放大」的平台缩放伪影（小红书图常被多次转码）。"""

    def __init__(self, p=0.3, scale=(0.5, 0.75)):
        self.p = p
        self.scale = scale

    def __call__(self, img):
        if random.random() < self.p:
            w, h = img.size
            f = random.uniform(self.scale[0], self.scale[1])
            small = img.resize((max(16, int(w * f)), max(16, int(h * f))), Image.BILINEAR)
            img = small.resize((w, h), Image.BILINEAR)
        return img


class RandomGaussianNoise:
    """模拟传感器/压缩噪声（轻度，避免过度破坏语义）。"""

    def __init__(self, p=0.25, std=(0.01, 0.04)):
        self.p = p
        self.std = std

    def __call__(self, img):
        if random.random() < self.p:
            arr = np.asarray(img).astype("float32")
            s = random.uniform(self.std[0], self.std[1]) * 255.0
            arr = arr + np.random.normal(0.0, s, arr.shape)
            arr = np.clip(arr, 0, 255).astype("uint8")
            return Image.fromarray(arr)
        return img


def get_train_transform():
    """训练增强：在原有基础上补强「真实美妆图分布」覆盖。

    新增项（相对旧 pipeline）：更强色彩抖动、小幅旋转、更强高斯模糊（近似磨皮）、
    平台缩放伪影、JPEG 重压缩、轻度噪声。全部只作用在自产图上。
    """
    return T.Compose([
        T.Resize(256),
        T.RandomCrop(IMG_SIZE),
        T.RandomHorizontalFlip(p=0.5),
        T.ColorJitter(0.4, 0.4, 0.4, 0.15),          # 滤镜/色调偏移
        T.RandomRotation(degrees=8),                 # 轻微拍摄倾斜
        T.RandomApply([T.GaussianBlur(kernel_size=5, sigma=(0.3, 2.5))], p=0.4),  # 近似磨皮
        RandomResizeNoise(p=0.3, scale=(0.5, 0.75)),  # 平台缩放伪影
        RandomJPEG(q=(55, 92)),                       # 重度 JPEG 重压缩
        RandomGaussianNoise(p=0.25, std=(0.01, 0.04)),  # 压缩/传感器噪声
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def get_infer_transform():
    """推理增强：与训练一致的预处理（不含随机项）。"""
    return T.Compose([
        T.Resize(256),
        T.CenterCrop(IMG_SIZE),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])
