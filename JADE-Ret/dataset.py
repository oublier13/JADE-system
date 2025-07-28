import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ClassificationDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, mask_transform=None):
        """
        Args:
            root_dir (str): 数据根目录，例如 'G:\\CODE\\datasets\\cla'
            csv_file (str): 包含图像相对路径和标签的文本文件，每行格式为 'path/to/image.jpg,label'
            transform: 图像变换
            mask_transform: mask的变换
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.samples = []

        # 手动读取CSV文件
        with open(csv_file, 'r', encoding='utf-8') as f:
            next(f)  # 跳过表头
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        self.samples.append((parts[0], int(parts[1])))  # (img_relative_path, label)


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_relative_path, label = self.samples[idx]
        img_path = os.path.join(self.root_dir, img_relative_path)
        mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 灰度图

        if self.transform:
            image = self.transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask, label
