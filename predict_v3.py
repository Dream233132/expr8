"""
PlantCLEF 2026 - V3 改进版预测脚本
关键改进：
1. 使用更强的模型 (ConvNeXt-Base)
2. 更智能的物种分配策略（基于聚类中心距离）
3. 动态预测物种数（根据图像复杂度）
4. 物种多样性约束（避免预测过于相似的物种）
"""

import torch
import torch.nn.functional as F
import os
import sys
import csv
import re
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict, Counter

# 解决编码问题
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("PlantCLEF 2026 - V3 Improved Prediction System")
print("=" * 70)

# ============================================================
# 配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

TEST_IMG_DIR = 'data/PlantCLEF2025_test_images/PlantCLEF2025_test_images'
METADATA_CSV = 'data/PlantCLEF2024_single_plant_training_metadata.csv'
SPECIES_IDS_CSV = 'data/species_ids.csv'
TEST_CSV = 'data/PlantCLEF2025_test.csv'
OUTPUT_CSV = 'submission_v3.csv'

# 动态物种数范围
MIN_SPECIES = 20
MAX_SPECIES = 40
DEFAULT_SPECIES = 30

print(f"\n配置:")
print(f"  - 物种数范围: {MIN_SPECIES}-{MAX_SPECIES} (默认{DEFAULT_SPECIES})")
print(f"  - 输出文件: {OUTPUT_CSV}")

# ============================================================
# 1. 加载元数据
# ============================================================
print("\n[1/7] 加载元数据...")

species_df = pd.read_csv(SPECIES_IDS_CSV)
valid_species = set(species_df['species_id'].tolist())
print(f"  有效物种数: {len(valid_species)}")

test_df = pd.read_csv(TEST_CSV, sep=';')
test_quadrats = test_df['quadrat_id'].tolist()
print(f"  测试样方数: {len(test_quadrats)}")

# 加载训练元数据统计物种频率和科属信息
print("  加载训练元数据...")
meta_df = pd.read_csv(METADATA_CSV, sep=';', usecols=['species_id', 'family', 'genus'])
species_counts = meta_df['species_id'].value_counts()
valid_species_counts = species_counts[species_counts.index.isin(valid_species)]

# 获取top 600候选物种
top_species_list = valid_species_counts.head(600).index.tolist()
print(f"  候选物种池: {len(top_species_list)}")

# 构建物种的科属信息字典
species_info = meta_df.drop_duplicates('species_id').set_index('species_id')
species_family = species_info['family'].to_dict()
species_genus = species_info['genus'].to_dict()
del meta_df

# ============================================================
# 2. 解析quadrat结构
# ============================================================
print("\n[2/7] 解析样方结构...")

quadrat_info = {}
site_quadrats = defaultdict(list)
region_quadrats = defaultdict(list)

for qid in test_quadrats:
    parts = qid.rsplit('-', 1)
    if len(parts) == 2:
        site = parts[0]
        date_str = parts[1]
    else:
        site = qid
        date_str = ""
    
    region_match = re.match(r'^([A-Za-z]+-[A-Za-z]+)', qid)
    region = region_match.group(1) if region_match else "unknown"
    
    month = int(date_str[4:6]) if len(date_str) >= 6 else 6
    season = 0 if month <= 3 else (1 if month <= 6 else (2 if month <= 9 else 3))
    
    quadrat_info[qid] = {'site': site, 'region': region, 'date': date_str, 'month': month, 'season': season}
    site_quadrats[site].append(qid)
    region_quadrats[region].append(qid)

print(f"  唯一站点数: {len(site_quadrats)}")
print(f"  唯一区域数: {len(region_quadrats)}")

# ============================================================
# 3. 加载模型并提取特征
# ============================================================
print("\n[3/7] 加载模型并提取特征...")

import timm
from torchvision import transforms as T

model_name = 'convnext_base.fb_in22k_ft_in1k'
print(f"  模型: {model_name}")

model = timm.create_model(model_name, pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

tta_transforms = [
    transform,
    T.Compose([T.Resize(320), T.CenterCrop(300), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(280), T.CenterCrop(256), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
]

print(f"  特征维度: {model.num_features}, TTA增强数: {len(tta_transforms)}")

features_dict = {}
img_files = [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith('.jpg')]
total = len(img_files)

with torch.no_grad():
    for i, img_name in enumerate(img_files):
        if (i + 1) % 300 == 0 or i == 0 or (i + 1) == total:
            print(f"  进度: {i+1}/{total}")
        
        img_path = os.path.join(TEST_IMG_DIR, img_name)
        quadrat_id = os.path.splitext(img_name)[0]
        
        try:
            image = Image.open(img_path).convert('RGB')
            feats = []
            for t in tta_transforms:
                img_tensor = t(image).unsqueeze(0).to(device)
                feat = model(img_tensor)
                feats.append(feat.cpu().numpy().flatten())
            features_dict[quadrat_id] = np.mean(feats, axis=0)
        except Exception as e:
            print(f"  警告: 跳过 {img_name}: {e}")

print(f"  已提取 {len(features_dict)} 张图片的特征")
