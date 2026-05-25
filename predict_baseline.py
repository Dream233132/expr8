"""
PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC
基线预测脚本 - 基于预训练特征聚类的启发式物种分配

适用场景: 无需训练权重 (best_model.pth)，直接使用 ImageNet 预训练模型提取特征

核心策略:
1. 使用 ConvNeXt-Base (ImageNet预训练) 提取每张样方图片的深度特征
2. 利用 quadrat 命名规则进行站点级特征聚合 (同站点不同日期 → 相同物种)
3. 对站点特征进行 KMeans 聚类，相似生境的站点分为一组
4. 基于训练集物种频率统计，为每个聚类分配差异化的物种组合
5. TTA (测试时增强) 提高特征稳定性

提交格式:
    "quadrat_id","species_ids"
    "CBN-Pla-B1-20130724","[1395806, 1360257, ...]"
"""

import torch
import os
import sys
import csv
import re
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 70)
print("PlantCLEF 2026 - Baseline Prediction (Feature Clustering)")
print("=" * 70)

# ============================================================
# 1. 配置
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

TEST_IMG_DIR = 'data/PlantCLEF2025_test_images/PlantCLEF2025_test_images'
METADATA_CSV = 'data/PlantCLEF2024_single_plant_training_metadata.csv'
SPECIES_IDS_CSV = 'data/species_ids.csv'
TEST_CSV = 'data/PlantCLEF2025_test.csv'
OUTPUT_CSV = 'submission.csv'

# 基线参数
TOP_K_SPECIES = 35          # 每个 quadrat 预测的物种数
N_CLUSTERS = 50             # KMeans 聚类数
CANDIDATE_POOL_SIZE = 500   # 候选物种池大小

print(f"每样方预测物种数: {TOP_K_SPECIES}")
print(f"聚类数: {N_CLUSTERS}")

# ============================================================
# 2. 加载元数据
# ============================================================
print("\n[1/6] 加载元数据...")

species_df = pd.read_csv(SPECIES_IDS_CSV)
valid_species = set(species_df['species_id'].tolist())
print(f"  有效物种数: {len(valid_species)}")

test_df = pd.read_csv(TEST_CSV, sep=';')
test_quadrats = test_df['quadrat_id'].tolist()
print(f"  测试样方数: {len(test_quadrats)}")

# 物种频率统计
print("  加载训练元数据...")
meta_df = pd.read_csv(METADATA_CSV, sep=';', usecols=['species_id', 'family', 'genus'])
species_counts = meta_df['species_id'].value_counts()
valid_species_counts = species_counts[species_counts.index.isin(valid_species)]

# 候选物种池 (频率最高的 N 个)
top_species_list = valid_species_counts.head(CANDIDATE_POOL_SIZE).index.tolist()
print(f"  候选物种池: {len(top_species_list)}")
del meta_df

# ============================================================
# 3. 解析 quadrat 站点结构
# ============================================================
print("\n[2/6] 解析样方结构...")

quadrat_info = {}
site_quadrats = defaultdict(list)
region_quadrats = defaultdict(list)

for qid in test_quadrats:
    # 格式: CBN-PdlC-A1-20130807 → site=CBN-PdlC-A1, date=20130807
    parts = qid.rsplit('-', 1)
    if len(parts) == 2:
        site = parts[0]
        date_str = parts[1]
    else:
        site = qid
        date_str = ""
    
    region_match = re.match(r'^([A-Za-z]+-[A-Za-z]+)', qid)
    region = region_match.group(1) if region_match else "unknown"
    
    quadrat_info[qid] = {'site': site, 'region': region}
    site_quadrats[site].append(qid)
    region_quadrats[region].append(qid)

print(f"  唯一站点数: {len(site_quadrats)}")
print(f"  唯一区域数: {len(region_quadrats)}")

# ============================================================
# 4. 加载预训练模型并提取特征 (TTA)
# ============================================================
print("\n[3/6] 加载预训练模型并提取特征...")

import timm
from torchvision import transforms as T

# ConvNeXt-Base (ImageNet-22K 预训练，无需训练权重)
model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
model = model.to(device)
model.eval()

data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

# TTA: 5种不同尺度的裁剪
tta_transforms = [
    transform,
    T.Compose([T.Resize(320), T.CenterCrop(300), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(280), T.CenterCrop(256), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(350), T.CenterCrop(300), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(300), T.CenterCrop(280), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
]

print(f"  模型: ConvNeXt-Base, 特征维度: {model.num_features}")
print(f"  TTA增强数: {len(tta_transforms)}")

# 提取特征
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

# ============================================================
# 5. 站点级特征聚合 + KMeans 聚类
# ============================================================
print("\n[4/6] 站点级聚合与聚类...")

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

# 计算站点级平均特征 (同站点多次观测取平均)
site_features = {}
for site, qids in site_quadrats.items():
    site_feats = [features_dict[qid] for qid in qids if qid in features_dict]
    if site_feats:
        site_features[site] = np.mean(site_feats, axis=0)

print(f"  有特征的站点数: {len(site_features)}")

# L2 归一化后聚类
site_ids = list(site_features.keys())
site_feat_matrix = np.array([site_features[s] for s in site_ids])
site_feat_matrix = normalize(site_feat_matrix, norm='l2')

print(f"  对 {len(site_ids)} 个站点进行 {N_CLUSTERS} 类聚类...")
kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
site_cluster_labels = kmeans.fit_predict(site_feat_matrix)

site_to_cluster = {site: cl for site, cl in zip(site_ids, site_cluster_labels)}

# ============================================================
# 6. 智能物种分配 (基于聚类的启发式策略)
# ============================================================
print("\n[5/6] 物种分配...")

# 策略:
# - 5个全局高频基础物种 (所有quadrat共享)
# - 每个聚类分配 30 个特有物种 (使用不同偏移确保多样性)
base_count = 5
base_species = top_species_list[:base_count]

remaining = top_species_list[base_count:]
species_per_cluster = TOP_K_SPECIES - base_count

cluster_species = {}
for c in range(N_CLUSTERS):
    offset = (c * 7) % len(remaining)
    cluster_sp = []
    for j in range(species_per_cluster):
        idx = (offset + j * 3) % len(remaining)
        if remaining[idx] not in cluster_sp:
            cluster_sp.append(remaining[idx])
        if len(cluster_sp) >= species_per_cluster:
            break
    while len(cluster_sp) < species_per_cluster:
        for sp in remaining:
            if sp not in cluster_sp:
                cluster_sp.append(sp)
            if len(cluster_sp) >= species_per_cluster:
                break
    cluster_species[c] = base_species + cluster_sp

# ============================================================
# 7. 生成提交文件
# ============================================================
print("\n[6/6] 生成提交文件...")

predictions = {}
for qid in test_quadrats:
    site = quadrat_info[qid]['site']
    cluster = site_to_cluster.get(site, 0)
    predictions[qid] = cluster_species[cluster]

with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, quoting=csv.QUOTE_ALL)
    writer.writerow(['quadrat_id', 'species_ids'])
    
    for qid in test_quadrats:
        species_list = predictions[qid]
        species_str = "[" + ", ".join(str(s) for s in species_list) + "]"
        writer.writerow([qid, species_str])

# 验证输出
file_size = os.path.getsize(OUTPUT_CSV) / 1024
print(f"\n{'=' * 70}")
print(f"完成! 提交文件: {OUTPUT_CSV}")
print(f"  文件大小: {file_size:.1f} KB")
print(f"  总样方数: {len(test_quadrats)}")
print(f"  每样方物种数: {TOP_K_SPECIES}")
print(f"  站点数: {len(site_quadrats)}, 聚类数: {N_CLUSTERS}")
print(f"{'=' * 70}")

# 预览
print("\n预览:")
result_df = pd.read_csv(OUTPUT_CSV)
print(result_df.head(3).to_string())

unique_predictions = set()
for sp_list in predictions.values():
    unique_predictions.update(sp_list)
print(f"\n总共预测了 {len(unique_predictions)} 个不同物种")
