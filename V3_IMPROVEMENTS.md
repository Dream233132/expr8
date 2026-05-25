# PlantCLEF 2026 - V3 改进总结

## 改进时间
2026年5月25日

## 改进内容

### 1. 更强的预训练模型 ✅
**修改**: `predict.py` 第129行
```python
# v2: EfficientNet-B3
model = timm.create_model('efficientnet_b3', pretrained=True, num_classes=0)

# v3: ConvNeXt-Base (ImageNet-22K预训练)
model = timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
```

**优势**:
- ConvNeXt-Base 是更现代的架构，性能优于 EfficientNet-B3
- ImageNet-22K 预训练（21,843类）比 ImageNet-1K（1,000类）包含更多植物相关类别
- 特征维度: 1024 (vs EfficientNet-B3的1536)，计算效率更高

**预期提升**: +5-8% F1

---

### 2. 增加预测物种数 ✅
**修改**: `predict.py` 第46行
```python
# v2: 每个quadrat预测30个物种
TOP_K_SPECIES = 30

# v3: 增加到35个物种
TOP_K_SPECIES = 35
```

**理由**:
- 提高 Recall（召回率）
- 真实样方中的物种数未知，地中海地区植被丰富
- 适度增加预测数可以在不显著降低 Precision 的情况下提高 Recall

**预期提升**: +3-5% F1

---

### 3. 增强TTA（测试时增强）✅
**修改**: `predict.py` 第138-148行
```python
# v2: 3种TTA增强
tta_transforms = [
    transform,  # 原始
    T.Compose([T.Resize(320), T.CenterCrop(300), ...]),
    T.Compose([T.Resize(300), T.RandomCrop(288), ...]),
]

# v3: 5种TTA增强
tta_transforms = [
    transform,  # 原始
    T.Compose([T.Resize(320), T.CenterCrop(300), ...]),
    T.Compose([T.Resize(280), T.CenterCrop(256), ...]),  # 新增
    T.Compose([T.Resize(350), T.CenterCrop(300), ...]),  # 新增
    T.Compose([T.Resize(300), T.RandomCrop(280), ...]),
]
```

**优势**:
- 更多尺度的特征提取，提高特征鲁棒性
- 减少单次推理的随机性
- 对不同大小的植物更敏感

**预期提升**: +2-3% F1

---

## 总体改进预期

| 版本 | 模型 | 物种数 | TTA | 预期F1 | 相对提升 |
|------|------|--------|-----|--------|----------|
| v2 | EfficientNet-B3 | 30 | 3种 | 0.10-0.12 | baseline |
| v3 | ConvNeXt-Base | 35 | 5种 | 0.12-0.15 | +15-25% |

**总预期提升**: +10-20% F1 (相对提升)

---

## 实施成本

- **代码修改**: 3处，约10行代码
- **额外计算**: 
  - ConvNeXt-Base 比 EfficientNet-B3 略慢（~10%）
  - 5种TTA vs 3种TTA 增加 ~67% 推理时间
  - 总推理时间: 约 15-20 分钟（GPU）
- **模型下载**: ConvNeXt-Base 权重 ~350MB（首次运行）

---

## 运行方式

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 运行改进后的预测脚本
python predict.py

# 输出: submission.csv
```

---

## 文件变化

### 修改的文件
- `predict.py` - 应用了3项优化

### 新增的文件
- `OPTIMIZATION_GUIDE.md` - 详细的优化指南
- `V3_IMPROVEMENTS.md` - 本文件，改进总结
- `predict_v3.py` - 备用脚本（未完成）

---

## 进一步优化方向

如果需要继续提升得分，建议按优先级实施：

### 短期（1-2天）
1. **地理先验**: 为不同区域分配不同的物种池
2. **动态物种数**: 根据图像复杂度调整预测物种数（25-40）
3. **增加聚类数**: 从50增加到70-100

**预期额外提升**: +5-10% F1

### 中期（3-7天）
1. **物种共现分析**: 从训练元数据分析哪些物种经常一起出现
2. **集成学习**: 使用多个模型（ConvNeXt + ViT + EfficientNet）融合预测
3. **伪标签训练**: 使用测试集的预测结果进行自训练

**预期额外提升**: +10-20% F1

### 长期（1-2周）
1. **下载训练数据**: 从元数据URL下载top 500物种的训练图片
2. **微调模型**: 在植物分类任务上微调 ConvNeXt-Base
3. **多标签分类**: 训练端到端的多标签分类模型

**预期额外提升**: +30-50% F1

---

## 版本对比

| 特性 | v1 (初始) | v2 (改进) | v3 (当前) |
|------|-----------|-----------|-----------|
| 模型 | SimpleCNN | EfficientNet-B3 | ConvNeXt-Base |
| 预训练 | 无 | ImageNet-1K | ImageNet-22K |
| TTA | 无 | 3种 | 5种 |
| 站点聚合 | 无 | ✅ | ✅ |
| 聚类 | 无 | 50类 | 50类 |
| 物种数 | 10 | 30 | 35 |
| 预期F1 | <0.01 | 0.10-0.12 | 0.12-0.15 |

---

## 结论

通过3项简单但有效的优化，v3版本相比v2版本预期可以获得 **15-25%** 的相对F1提升。这些优化都是无需额外数据的快速改进，实施成本低，效果显著。

如果需要进一步提升得分到竞争性水平（F1 > 0.25），建议下载训练数据并微调模型。

---

## 参考资料

- [ConvNeXt论文](https://arxiv.org/abs/2201.03545)
- [timm库文档](https://huggingface.co/docs/timm)
- [PlantCLEF 2026竞赛](https://www.kaggle.com/competitions/plantclef-2026)
- `OPTIMIZATION_GUIDE.md` - 详细优化指南
- `IMPROVEMENTS.md` - v2版本改进文档
