"""
PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC
主训练脚本 - 基于 ConvNeXt-Base 的多标签植物分类

核心设计:
1. 数据加载: 读取训练元数据CSV，构建 species_id → index 映射，返回 multi-hot 标签
2. 模型架构: ConvNeXt-Base (ImageNet预训练) + 7806维分类头
3. 损失函数: Asymmetric Loss (ASL) - 专为多标签分类设计
4. 训练优化: 混合精度训练 (AMP) 适配 RTX 3050 8G 显存
5. 权重保存: 每个 Epoch 保存 Loss 最低的权重为 best_model.pth
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms, models
from PIL import Image
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# 解决 Windows 终端编码问题
# ============================================================
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# ============================================================
# 1. 全局配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 路径配置
METADATA_CSV = 'data/PlantCLEF2024_single_plant_training_metadata.csv'
SPECIES_IDS_CSV = 'data/species_ids.csv'
TRAIN_IMG_DIR = 'data/train'  # 训练图片目录: data/train/{species_id}/{image_name}.jpg

# 训练超参数
NUM_CLASSES = 7806          # 物种总数
BATCH_SIZE = 16             # 适配 RTX 3050 8G 显存
NUM_EPOCHS = 20             # 训练轮数
LEARNING_RATE = 1e-4        # 学习率
WEIGHT_DECAY = 1e-4         # 权重衰减
NUM_WORKERS = 4             # 数据加载线程数
IMG_SIZE = 224              # 输入图像尺寸

print("=" * 70)
print("PlantCLEF 2026 - Training Pipeline (ConvNeXt-Base + ASL)")
print("=" * 70)
print(f"设备: {device}")
print(f"批大小: {BATCH_SIZE}, 学习率: {LEARNING_RATE}, 轮数: {NUM_EPOCHS}")
print(f"图像尺寸: {IMG_SIZE}x{IMG_SIZE}, 类别数: {NUM_CLASSES}")


# ============================================================
# 2. 构建 species_id → index 映射
# ============================================================
print("\n[1/5] 构建物种ID映射...")

species_df = pd.read_csv(SPECIES_IDS_CSV)
species_list = species_df['species_id'].tolist()  # 长度 7806
# 构建双向映射
species_to_idx = {sp: idx for idx, sp in enumerate(species_list)}
idx_to_species = {idx: sp for idx, sp in enumerate(species_list)}

print(f"  物种总数: {len(species_list)}")
print(f"  ID范围: {min(species_list)} ~ {max(species_list)}")


# ============================================================
# 3. 数据集类
# ============================================================
class PlantCLEFDataset(Dataset):
    """
    PlantCLEF 2024 训练数据集
    
    目录结构: data/train/{species_id}/{image_name}.jpg
    标签: 7806维 multi-hot 浮点张量 (单标签时只有一个位置为1.0)
    """
    
    def __init__(self, metadata_csv, img_dir, species_to_idx, transform=None, max_samples=None):
        """
        Args:
            metadata_csv: 训练元数据CSV路径
            img_dir: 训练图片根目录
            species_to_idx: species_id → index 映射字典
            transform: 图像变换
            max_samples: 最大样本数 (用于调试)
        """
        self.img_dir = img_dir
        self.species_to_idx = species_to_idx
        self.num_classes = len(species_to_idx)
        self.transform = transform
        
        # 读取元数据
        print("  读取训练元数据...")
        df = pd.read_csv(metadata_csv, sep=';', usecols=['image_name', 'species_id'])
        
        # 只保留有效物种
        df = df[df['species_id'].isin(species_to_idx.keys())]
        
        if max_samples:
            df = df.head(max_samples)
        
        # 构建样本列表: (image_path, species_id)
        self.samples = []
        for _, row in df.iterrows():
            img_name = row['image_name']
            sp_id = int(row['species_id'])
            # 图片路径: data/train/{species_id}/{image_name}
            img_path = os.path.join(img_dir, str(sp_id), img_name)
            self.samples.append((img_path, sp_id))
        
        print(f"  有效样本数: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, species_id = self.samples[idx]
        
        # 加载图像
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError):
            # 如果图片不存在，返回随机噪声图像（训练时跳过）
            image = Image.new('RGB', (IMG_SIZE, IMG_SIZE), color=(128, 128, 128))
        
        if self.transform:
            image = self.transform(image)
        
        # 构建 multi-hot 标签向量 (7806维)
        label = torch.zeros(self.num_classes, dtype=torch.float32)
        idx_in_label = self.species_to_idx[species_id]
        label[idx_in_label] = 1.0
        
        return image, label


# ============================================================
# 4. Asymmetric Loss (ASL) - 多标签分类专用损失函数
# ============================================================
class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for Multi-Label Classification
    论文: https://arxiv.org/abs/2009.14119
    
    核心思想:
    - 对正样本(标签=1)使用较小的 gamma_pos 进行 focusing
    - 对负样本(标签=0)使用较大的 gamma_neg 进行 focusing + 概率裁剪
    - 这样可以有效抑制大量负样本的梯度贡献，聚焦于困难正样本
    
    Args:
        gamma_neg: 负样本的 focusing 参数 (越大越抑制简单负样本)
        gamma_pos: 正样本的 focusing 参数
        clip: 负样本概率裁剪阈值 (将低概率负样本的贡献截断)
    """
    
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
    
    def forward(self, logits, targets):
        """
        Args:
            logits: 模型原始输出 [batch_size, num_classes], 未经sigmoid
            targets: multi-hot标签 [batch_size, num_classes], 值为0或1
        Returns:
            loss: 标量损失值
        """
        # 计算概率
        probs = torch.sigmoid(logits)
        
        # 分离正负样本的概率
        probs_pos = probs   # 正样本使用原始概率
        probs_neg = probs   # 负样本概率
        
        # 负样本概率裁剪 (Probability Margin)
        # 将负样本的概率向下偏移，使得低概率负样本的loss更小
        if self.clip > 0:
            probs_neg = (probs_neg + self.clip).clamp(max=1.0)
        
        # 计算正负样本的交叉熵
        # 正样本: -log(p)
        loss_pos = targets * torch.log(probs_pos.clamp(min=self.eps))
        # 负样本: -log(1-p)
        loss_neg = (1 - targets) * torch.log((1 - probs_neg).clamp(min=self.eps))
        
        loss = loss_pos + loss_neg
        
        # Asymmetric Focusing
        # 正样本: (1-p)^gamma_pos 作为权重
        if self.gamma_pos > 0:
            pt_pos = probs_pos * targets + (1 - probs_pos) * (1 - targets)
            pos_weight = torch.pow(1 - pt_pos, self.gamma_pos)
            # 只对正样本部分应用 focusing
            pos_mask = (targets == 1).float()
            loss = loss * (1 - pos_mask) + loss_pos * pos_weight * pos_mask
        
        # 负样本: p^gamma_neg 作为权重 (抑制简单负样本)
        if self.gamma_neg > 0:
            neg_mask = (targets == 0).float()
            neg_weight = torch.pow(probs_neg.detach(), self.gamma_neg)
            loss = loss * (1 - neg_mask) + loss_neg * neg_weight * neg_mask
        
        return -loss.sum() / logits.size(0)


# ============================================================
# 5. 模型定义
# ============================================================
def build_model(num_classes=NUM_CLASSES, pretrained=True):
    """
    构建 ConvNeXt-Base 分类模型
    
    架构: ConvNeXt-Base (ImageNet预训练) + 自定义分类头
    输出: num_classes 维 logits (未经sigmoid)
    """
    # 加载预训练的 ConvNeXt-Base
    model = models.convnext_base(
        weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
    )
    
    # 获取原始分类头的输入维度 (ConvNeXt-Base: 1024)
    in_features = model.classifier[2].in_features
    
    # 替换分类头: LayerNorm + Flatten + Linear(1024, 7806)
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    print(f"  模型: ConvNeXt-Base")
    print(f"  分类头: Linear({in_features}, {num_classes})")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    return model


# ============================================================
# 6. 训练主流程
# ============================================================
if __name__ == "__main__":
    
    # ----------------------------------------------------------
    # 6.1 数据预处理 (训练时使用数据增强)
    # ----------------------------------------------------------
    print("\n[2/5] 配置数据增强...")
    
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),  # CutOut 增强
    ])
    
    print("  增强策略: Flip + Rotation + ColorJitter + Affine + RandomErasing")
    
    # ----------------------------------------------------------
    # 6.2 创建数据集和数据加载器
    # ----------------------------------------------------------
    print("\n[3/5] 加载训练数据集...")
    
    # 检查训练图片目录是否存在
    if not os.path.exists(TRAIN_IMG_DIR):
        print(f"\n{'!'*70}")
        print(f"  警告: 训练图片目录 '{TRAIN_IMG_DIR}' 不存在!")
        print(f"  请先下载训练图片到该目录，目录结构:")
        print(f"    data/train/{{species_id}}/{{image_name}}.jpg")
        print(f"  例如: data/train/1396710/59feabe1c98f06e7f819f73c8246bd8f1a89556b.jpg")
        print(f"{'!'*70}")
        print(f"\n  你可以从元数据CSV中的 image_backup_url 列下载图片。")
        print(f"  退出训练。")
        sys.exit(1)
    
    train_dataset = PlantCLEFDataset(
        metadata_csv=METADATA_CSV,
        img_dir=TRAIN_IMG_DIR,
        species_to_idx=species_to_idx,
        transform=train_transform,
        max_samples=None  # 设为整数可限制样本数用于调试，如 max_samples=10000
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )
    
    print(f"  数据加载器: {len(train_loader)} 个batch/epoch")
    
    # ----------------------------------------------------------
    # 6.3 初始化模型、损失函数、优化器
    # ----------------------------------------------------------
    print("\n[4/5] 初始化模型...")
    
    model = build_model(num_classes=NUM_CLASSES, pretrained=True).to(device)
    
    # ASL 损失函数
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)
    print(f"  损失函数: AsymmetricLoss(γ_neg=4, γ_pos=1, clip=0.05)")
    
    # AdamW 优化器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    
    # 混合精度训练的 GradScaler
    scaler = GradScaler()
    
    print(f"  优化器: AdamW(lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
    print(f"  调度器: CosineAnnealing(T_max={NUM_EPOCHS})")
    print(f"  混合精度: 已启用 (AMP + GradScaler)")
    
    # ----------------------------------------------------------
    # 6.4 训练循环
    # ----------------------------------------------------------
    print(f"\n[5/5] 开始训练 ({NUM_EPOCHS} epochs)...")
    print("-" * 70)
    
    best_loss = float('inf')
    loss_history = []
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        num_batches = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # 混合精度前向传播
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            
            # 混合精度反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            num_batches += 1
            
            # 每100个batch打印一次进度
            if (batch_idx + 1) % 100 == 0:
                avg = running_loss / num_batches
                print(f"  Epoch {epoch+1}/{NUM_EPOCHS} | "
                      f"Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {avg:.4f}")
        
        # Epoch 结束
        scheduler.step()
        epoch_loss = running_loss / num_batches
        loss_history.append(epoch_loss)
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"  ★ Epoch [{epoch+1}/{NUM_EPOCHS}] 完成 | "
              f"Loss: {epoch_loss:.4f} | LR: {current_lr:.2e}")
        
        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"    → 保存最佳模型 (loss={best_loss:.4f})")
    
    print("-" * 70)
    print(f"\n训练完成! 最佳 Loss: {best_loss:.4f}")
    print(f"模型权重已保存: best_model.pth")
    
    # ----------------------------------------------------------
    # 6.5 绘制 Loss 曲线
    # ----------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), loss_history, 'b-o', linewidth=2, markersize=6)
    plt.title('PlantCLEF 2026 - Training Loss (ConvNeXt-Base + ASL)', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=150)
    print(f"Loss 曲线已保存: loss_curve.png")
