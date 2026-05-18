# PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC

## 实验八：植物物种识别与生物多样性评估

### 1. 项目简介

本项目参加 Kaggle 竞赛 **PlantCLEF 2026**，目标是从样方（quadrat）照片中预测存在的植物物种。每张测试图片是一个 50×50cm 的地面植被样方照片，需要识别其中所有可见的植物物种。

**竞赛链接**: [PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC](https://www.kaggle.com/competitions/plantclef-2026)

### 2. 评分公式

竞赛使用 **宏平均 F1 分数** 作为评估指标：

$$Score = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{j=1}^{T_i} F1^j$$

其中：
- $N$ 是样线（transect）的总数
- $T_i$ 是每条样线中的样方数
- $F1^j$ 是每个测试图像的宏平均 F1 得分

单个图像的 F1 计算：
$$F1^j = \frac{2 \cdot Precision_j \cdot Recall_j}{Precision_j + Recall_j}$$

- **Precision** = TP / (TP + FP)：预测正确的物种数 / 总预测物种数
- **Recall** = TP / (TP + FN)：预测正确的物种数 / 实际存在的物种数

### 3. 数据集结构

```
data/
├── PlantCLEF2024_single_plant_training_metadata.csv  # 训练集元数据 (140万条记录, 7806个物种)
├── PlantCLEF2025_test.csv                            # 测试集 quadrat 列表 (2105个样方)
├── species_ids.csv                                    # 有效物种 ID 列表 (7806个)
├── pseudoquadrats_without_labels_complementary_training_set_urls.csv  # 补充训练集URL
└── PlantCLEF2025_test_images/
    └── PlantCLEF2025_test_images/                    # 2105张测试样方照片 (.jpg)
```

### 4. 提交格式

CSV 文件，包含两列：
```csv
"quadrat_id","species_ids"
"CBN-Pla-B1-20130724","[1395806]"
"CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535, 1417626, 1295807]"
```

### 5. 解决方案

#### 5.1 技术方案概述

由于训练图片需要从远程URL下载（约140万张），本方案采用**零样本预测 + 预训练特征聚类**策略：

1. **预训练模型特征提取**：使用 EfficientNet-B3（ImageNet 预训练）提取每张样方图片的深度特征向量
2. **测试时增强（TTA）**：对每张图片使用3种不同的裁剪/缩放方式，取特征平均值提高稳定性
3. **站点级特征聚合**：利用 quadrat 命名规则（同一站点不同日期），将同站点的特征取平均
4. **精细聚类**：对656个站点的聚合特征进行 KMeans 聚类（50类），相似生境的站点分为一组
5. **差异化物种分配**：基于训练集物种频率统计，为每个聚类分配不同的物种组合

#### 5.2 关键改进点

| 改进项 | 基线方案 | 改进方案 |
|--------|----------|----------|
| 模型 | SimpleCNN (随机权重) | EfficientNet-B3 (ImageNet预训练) |
| 图像处理 | 128×128 单一变换 | 300px 多尺度 TTA |
| 特征聚合 | 无 | 站点级时序聚合 |
| 聚类 | 20类 | 50类精细聚类 |
| 物种分配 | 固定 top-K | 聚类差异化分配 |
| 预测物种数 | 25 | 30 (平衡 Precision/Recall) |

#### 5.3 quadrat 结构分析

测试集 quadrat 命名格式: `{区域}-{站点}-{日期}`

| 区域 | 数量 | 说明 |
|------|------|------|
| CBN-PdlC | 816 | 法国地中海植物保护区 |
| CBN-Pla | 628 | 法国植物保护区 |
| GUARDEN-CBNMed | 165 | 地中海地区监测 |
| LISAH-BOU/BVD/PEC/JAS | 208 | LISAH 观测站 |
| CBN-can | 30 | 加那利群岛 |
| RNNB-* | 141 | 自然保护区网络 |
| OPTMix-* | ~50 | 混合林监测 |

### 6. 运行方式

#### 环境要求

- Python 3.11+
- PyTorch 2.7+ (CUDA 11.8)
- timm 1.0+
- scikit-learn 1.8+
- pandas, numpy, Pillow

#### 运行预测

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 运行预测脚本，生成 submission.csv
python predict.py
```

#### 运行训练（演示）

```bash
python main.py
```

### 7. 文件说明

| 文件 | 说明 |
|------|------|
| `predict.py` | 预测脚本，生成 Kaggle 提交文件 |
| `main.py` | 模型训练流程（基于 ResNet50 微调） |
| `test.py` | GPU 环境检测脚本 |
| `submission.csv` | 生成的 Kaggle 提交文件 |
| `loss_curve.png` | 训练损失曲线 |

### 8. 进一步改进方向

1. **下载训练图片进行微调**：从元数据中的 URL 下载训练图片，对 EfficientNet-B3 进行植物物种分类微调
2. **使用植物专用模型**：如 BioCLIP、PlantNet 预训练模型
3. **物种共现建模**：分析同一 quadrat 中哪些物种经常同时出现
4. **地理先验**：结合 quadrat 的地理位置信息（经纬度、海拔）过滤不可能出现的物种
5. **集成学习**：融合多个模型的预测结果

### 9. 参考资料

- [PlantCLEF 2026 竞赛页面](https://www.kaggle.com/competitions/plantclef-2026)
- [timm 文档](https://huggingface.co/docs/timm)
- [scikit-learn KMeans](https://scikit-learn.org/stable/modules/clustering.html)
- F1 分数计算参考: scikit-learn 文档
