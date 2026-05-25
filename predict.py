"""
PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC
预测脚本 - 加载训练好的 ConvNeXt-Base 模型进行多标签推理

核心流程:
1. 加载 best_model.pth 权重 (与 main.py 相同的 ConvNeXt-Base 架构)
2. 对测试集样方图像进行推理，输出经过 sigmoid 的概率
3. 阈值截断: THRESHOLD=0.15，找出所有概率大于阈值的类别
4. Fallback: 如果所有概率都低于阈值，保留概率最高的 1 个预测
5. 将内部 index 映射回真实 species_id，导出 submission.csv

提交格式:
    "quadrat_id","species_ids"
    "CBN-Pla-B1-20130724","[1395806]"
    "CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367]"
"""

import torch
import torch.nn as nn
import os
import sys
import csv
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms, models

# ============================================================
# 解决 Windows 终端编码问题
# ============================================================
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("PlantCLEF 2026 - Prediction Pipeline (ConvNeXt-Base)")
print("=" * 70)

# ============================================================
# 1. 配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"推理设备: {device}")

# 路径配置
TEST_IMG_DIR = 'data/PlantCLEF2025_test_images/PlantCLEF2025_test_images'
SPECIES_IDS_CSV = 'data/species_ids.csv'
TEST_CSV = 'data/PlantCLEF2025_test.csv'
MODEL_PATH = 'best_model.pth'
OUTPUT_CSV = 'submission.csv'

# 推理参数
NUM_CLASSES = 7806
IMG_SIZE = 224
THRESHOLD = 0.15  # sigmoid 概率阈值

print(f"阈值: {THRESHOLD}")
print(f"模型权重: {MODEL_PATH}")
print(f"输出文件: {OUTPUT_CSV}")

# ============================================================
# 2. 构建 index → species_id 映射
# ============================================================
print("\n[1/4] 构建物种ID映射...")

species_df = pd.read_csv(SPECIES_IDS_CSV)
species_list = species_df['species_id'].tolist()  # 长度 7806
idx_to_species = {idx: sp for idx, sp in enumerate(species_list)}

print(f"  物种总数: {len(species_list)}")

# 加载测试集 quadrat 列表
test_df = pd.read_csv(TEST_CSV, sep=';')
test_quadrats = test_df['quadrat_id'].tolist()
print(f"  测试样方数: {len(test_quadrats)}")

# ============================================================
# 3. 构建模型并加载权重
# ============================================================
print("\n[2/4] 加载模型...")

def build_model(num_classes=NUM_CLASSES):
    """构建与训练时相同的 ConvNeXt-Base 架构"""
    model = models.convnext_base(weights=None)  # 不加载预训练权重
    in_features = model.classifier[2].in_features  # 1024
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

model = build_model(NUM_CLASSES).to(device)

# 加载训练好的权重
if os.path.exists(MODEL_PATH):
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"  ✓ 成功加载权重: {MODEL_PATH}")
else:
    print(f"  ✗ 权重文件不存在: {MODEL_PATH}")
    print(f"    请先运行 main.py 训练模型，或将权重文件放到当前目录。")
    print(f"    将使用随机初始化的模型进行推理（结果无意义，仅验证流程）。")

model.eval()
print(f"  模型: ConvNeXt-Base, 输出维度: {NUM_CLASSES}")

# ============================================================
# 4. 图像预处理 (测试时不使用数据增强)
# ============================================================
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ============================================================
# 5. 推理循环
# ============================================================
print("\n[3/4] 开始推理...")

predictions = {}  # quadrat_id → [species_id, ...]
total = len(test_quadrats)
fallback_count = 0  # 统计 fallback 次数

with torch.no_grad():
    for i, quadrat_id in enumerate(test_quadrats):
        # 进度显示
        if (i + 1) % 300 == 0 or i == 0 or (i + 1) == total:
            print(f"  进度: {i+1}/{total} ({(i+1)/total*100:.1f}%)")
        
        # 构建图片路径
        img_path = os.path.join(TEST_IMG_DIR, f"{quadrat_id}.jpg")
        
        # 加载并预处理图像
        try:
            image = Image.open(img_path).convert('RGB')
        except (FileNotFoundError, OSError) as e:
            print(f"  警告: 无法加载 {img_path}: {e}")
            # 如果图片缺失，使用 fallback 预测（频率最高的物种）
            predictions[quadrat_id] = [species_list[0]]
            fallback_count += 1
            continue
        
        # 预处理
        img_tensor = test_transform(image).unsqueeze(0).to(device)
        
        # 模型推理
        logits = model(img_tensor)  # [1, 7806]
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()  # [7806]
        
        # 阈值截断: 找出所有概率 > THRESHOLD 的类别
        above_threshold = np.where(probs > THRESHOLD)[0]
        
        if len(above_threshold) > 0:
            # 正常情况: 有类别超过阈值
            # 按概率从高到低排序
            sorted_indices = above_threshold[np.argsort(probs[above_threshold])[::-1]]
            predicted_species = [idx_to_species[idx] for idx in sorted_indices]
        else:
            # Fallback: 所有概率都低于阈值，取概率最高的 1 个
            top1_idx = np.argmax(probs)
            predicted_species = [idx_to_species[top1_idx]]
            fallback_count += 1
        
        predictions[quadrat_id] = predicted_species

print(f"  推理完成!")
print(f"  Fallback 次数: {fallback_count}/{total} ({fallback_count/total*100:.1f}%)")

# 统计预测物种数分布
pred_counts = [len(v) for v in predictions.values()]
print(f"  平均预测物种数: {np.mean(pred_counts):.1f}")
print(f"  最少: {min(pred_counts)}, 最多: {max(pred_counts)}")

# ============================================================
# 6. 导出 submission.csv
# ============================================================
print("\n[4/4] 导出提交文件...")

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['quadrat_id', 'species_ids'])
    
    for quadrat_id in test_quadrats:
        species_ids = predictions[quadrat_id]
        # 格式: "[ID1, ID2, ID3]"
        species_str = "[" + ", ".join(str(s) for s in species_ids) + "]"
        writer.writerow([quadrat_id, species_str])

# 验证输出
file_size = os.path.getsize(OUTPUT_CSV) / 1024
print(f"\n{'=' * 70}")
print(f"完成! 提交文件: {OUTPUT_CSV}")
print(f"  文件大小: {file_size:.1f} KB")
print(f"  总行数: {total + 1} (含表头)")
print(f"  格式: csv.QUOTE_ALL (全双引号)")
print(f"{'=' * 70}")

# 预览前3行
print("\n预览:")
result_df = pd.read_csv(OUTPUT_CSV)
print(result_df.head(3).to_string())

# 统计唯一物种数
all_species = set()
for sp_list in predictions.values():
    all_species.update(sp_list)
print(f"\n总共预测了 {len(all_species)} 个不同物种")
