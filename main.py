"""
PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC
主程序 - 模型训练流程

评分公式: 宏平均 F1 分数
  Score = (1/N) * sum_i( (1/T_i) * sum_j(F1_j) )
  其中 F1_j = 2 * Precision_j * Recall_j / (Precision_j + Recall_j)

提交格式: CSV 文件
  "quadrat_id","species_ids"
  "CBN-Pla-B1-20130724","[1395806]"
  "CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535, 1417626, 1295807]"
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import os
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# 1. 配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{'='*60}")
print(f"PlantCLEF 2026 Training Pipeline")
print(f"Device: {device}")
print(f"{'='*60}")

# 数据路径
TEST_IMG_DIR = 'data/PlantCLEF2025_test_images/PlantCLEF2025_test_images'
METADATA_CSV = 'data/PlantCLEF2024_single_plant_training_metadata.csv'
SPECIES_IDS_CSV = 'data/species_ids.csv'

# ============================================================
# 2. 数据预处理
# ============================================================
# 训练时使用数据增强
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证/测试时不使用数据增强
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ============================================================
# 3. 数据集类
# ============================================================
class PlantDataset(Dataset):
    """植物分类数据集，支持从元数据CSV加载"""
    
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.img_paths = []
        
        # 递归扫描所有图片
        for root, dirs, files in os.walk(img_dir):
            for f in files:
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    self.img_paths.append(os.path.join(root, f))
        
        if len(self.img_paths) == 0:
            raise FileNotFoundError(
                f"No images found in {img_dir}"
            )
        
        # 注意：这里使用文件名hash作为伪标签仅用于演示
        # 真正训练时应该从训练元数据中加载species_id标签
        self.labels = [hash(os.path.basename(p)) % 10 for p in self.img_paths]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        return image, label

# ============================================================
# 4. 模型定义 (基于预训练 ResNet50 微调)
# ============================================================
class PlantClassifier(nn.Module):
    """基于 ResNet50 的植物分类器"""
    
    def __init__(self, num_classes=7806):
        super(PlantClassifier, self).__init__()
        # 加载预训练的 ResNet50
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        
        # 替换最后的全连接层
        in_features = self.backbone.fc.in_features  # 2048
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


# ============================================================
# 5. 训练流程
# ============================================================
if __name__ == "__main__":
    # 加载数据
    print("\nLoading dataset...")
    full_dataset = PlantDataset(img_dir=TEST_IMG_DIR, transform=train_transform)
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    print(f"  Total images: {len(full_dataset)}")
    print(f"  Train: {train_size}, Val: {val_size}")

    # 初始化模型
    # 注意：这里用10类做演示，完整版应该用7806类
    model = PlantClassifier(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # 训练
    epochs = 5
    loss_history = []
    
    print(f"\nStarting training ({epochs} epochs)...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        scheduler.step()
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"  Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")

    # 保存 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, epochs+1), loss_history, marker='o', linestyle='-', 
             color='b', label='Training Loss')
    plt.title('PlantCLEF 2026 - Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('loss_curve.png')
    print(f"\nTraining complete! Loss curve saved to loss_curve.png")
