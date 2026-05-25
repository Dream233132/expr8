# PlantCLEF 2026 优化指南

## 当前方案总结 (v2)

### 已实现的优点 ✅
1. **预训练模型**: EfficientNet-B3 (ImageNet预训练)
2. **TTA增强**: 3种不同裁剪方式的特征平均
3. **站点级聚合**: 同站点多次观测特征取平均 (2105→656)
4. **精细聚类**: 50类KMeans聚类
5. **差异化物种分配**: 每个聚类不同的物种组合
6. **预测物种数**: 每个quadrat预测30个物种

### 当前得分预估
- 基于零样本预测 + 频率先验
- 预期F1分数: **0.05-0.15** (相对于随机baseline有显著提升)

---

## 进一步优化方案

### 🎯 优化方向1: 更强的预训练模型

**当前**: EfficientNet-B3 (ImageNet-1K/21K)

**改进选项**:
```python
# 选项A: ConvNeXt-Base (更强的架构)
model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0)

# 选项B: ViT-Base (Transformer架构)
model = timm.create_model('vit_base_patch16_224.augreg_in21k_ft_in1k', pretrained=True, num_classes=0)

# 选项C: DINOv2 (自监督学习，对植物特征更敏感)
model = timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=True, num_classes=0)
```

**预期提升**: +5-10% F1

---

### 🎯 优化方向2: 动态物种数预测

**当前问题**: 所有quadrat都预测固定30个物种

**改进方案**:
```python
# 根据图像复杂度动态调整
def estimate_species_count(feature_vector, cluster_id):
    """
    基于特征的标准差/熵估计物种丰富度
    """
    feature_std = np.std(feature_vector)
    
    # 特征方差大 → 图像复杂 → 物种多
    if feature_std > threshold_high:
        return 35  # 复杂场景
    elif feature_std > threshold_low:
        return 30  # 中等复杂度
    else:
        return 25  # 简单场景
```

**预期提升**: +3-5% F1

---

### 🎯 优化方向3: 物种共现模式

**当前问题**: 物种分配基于频率，忽略了生态学共现关系

**改进方案**:
```python
# 从训练元数据分析物种共现
# (需要下载训练图片或使用伪样方数据)

# 1. 构建物种共现矩阵
cooccurrence_matrix = analyze_training_data()

# 2. 预测时考虑共现约束
def select_species_with_cooccurrence(base_species, candidate_pool, cooccurrence_matrix, k=30):
    """
    选择与base_species共现概率高的物种
    """
    selected = list(base_species)
    
    for _ in range(k - len(selected)):
        scores = []
        for sp in candidate_pool:
            if sp in selected:
                continue
            # 计算与已选物种的平均共现分数
            score = np.mean([cooccurrence_matrix[sp][s] for s in selected])
            scores.append((sp, score))
        
        # 选择共现分数最高的
        best_sp = max(scores, key=lambda x: x[1])[0]
        selected.append(best_sp)
    
    return selected
```

**预期提升**: +10-15% F1

---

### 🎯 优化方向4: 地理先验

**当前问题**: 未利用测试集的地理区域信息

**改进方案**:
```python
# 为不同地理区域分配不同的物种池
region_species_pools = {
    'CBN-PdlC': top_species_mediterranean[:400],  # 地中海地区
    'CBN-Pla': top_species_temperate[:400],       # 温带地区
    'GUARDEN-CBNMed': top_species_mediterranean[:400],
    'LISAH': top_species_agricultural[:400],      # 农业区
    # ...
}

# 预测时只从对应区域的物种池中选择
def predict_with_geographic_prior(quadrat_id, cluster_id):
    region = extract_region(quadrat_id)
    species_pool = region_species_pools.get(region, top_species_list)
    return select_from_pool(species_pool, cluster_id, k=30)
```

**预期提升**: +5-8% F1

---

### 🎯 优化方向5: 集成学习

**方案**: 训练/使用多个模型并融合预测

```python
models = [
    'efficientnet_b3',
    'convnext_base.fb_in22k_ft_in1k',
    'vit_base_patch16_224.augreg_in21k_ft_in1k',
]

# 提取多模型特征并拼接
features_ensemble = []
for model_name in models:
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    feat = extract_features(model, image)
    features_ensemble.append(feat)

# 拼接特征
combined_features = np.concatenate(features_ensemble, axis=1)

# 基于组合特征进行聚类和预测
```

**预期提升**: +8-12% F1

---

### 🎯 优化方向6: 微调模型 (最大提升)

**当前限制**: 未下载训练图片，无法微调

**如果下载训练数据**:
```python
# 1. 下载top 500物种的训练图片 (约10-20GB)
# 2. 构建多标签分类数据集
# 3. 微调EfficientNet-B3

model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=7806)

# 使用BCEWithLogitsLoss进行多标签训练
criterion = nn.BCEWithLogitsLoss()

# 训练10-20 epochs
# ...

# 预测时使用sigmoid输出top-K物种
outputs = torch.sigmoid(model(image))
top_k_species = torch.topk(outputs, k=30).indices
```

**预期提升**: +30-50% F1 (最大提升)

---

## 快速实施建议

### 方案A: 无需额外数据 (1-2小时)
1. ✅ 替换为ConvNeXt-Base模型
2. ✅ 实现动态物种数
3. ✅ 添加地理先验

**预期总提升**: +15-20% F1

### 方案B: 使用伪样方数据 (3-5小时)
1. 下载 `pseudoquadrats_without_labels_complementary_training_set_urls.csv` 中的图片
2. 分析物种共现模式
3. 实现共现约束的物种选择

**预期总提升**: +25-35% F1

### 方案C: 完整训练 (1-2天)
1. 下载完整训练集 (140万图片)
2. 微调多标签分类模型
3. 集成多个模型

**预期总提升**: +50-80% F1

---

## 代码修改示例

### 修改1: 替换为ConvNeXt-Base

在 `predict.py` 第129行:
```python
# 修改前
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)

# 修改后
model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
```

### 修改2: 增加预测物种数

在 `predict.py` 第46行:
```python
# 修改前
TOP_K_SPECIES = 30

# 修改后
TOP_K_SPECIES = 35  # 提高Recall
```

### 修改3: 更多TTA增强

在 `predict.py` 第138-144行:
```python
# 添加更多增强
tta_transforms = [
    transform,
    T.Compose([T.Resize(320), T.CenterCrop(300), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(280), T.CenterCrop(256), T.ToTensor(),
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(350), T.CenterCrop(300), T.ToTensor(),  # 新增
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
    T.Compose([T.Resize(300), T.RandomCrop(280), T.ToTensor(),  # 新增
               T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]),
]
```

---

## 总结

当前v2方案已经是一个相当不错的零样本baseline。要进一步提升得分，建议按优先级实施：

1. **立即可做** (无需额外数据):
   - 替换ConvNeXt-Base模型
   - 增加物种数到35
   - 添加更多TTA增强
   
2. **中期优化** (需要少量数据):
   - 分析物种共现模式
   - 实现地理先验
   
3. **长期目标** (需要完整训练):
   - 下载训练数据微调模型
   - 实现集成学习

**预期最终F1分数**:
- 当前v2: ~0.10
- 快速优化后: ~0.12-0.15
- 使用伪样方数据: ~0.15-0.20
- 完整微调: ~0.25-0.35+
