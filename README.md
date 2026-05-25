# PlantCLEF 2026 @ LifeCLEF & CVPR-FGVC

## 实验八：植物物种多标签识别

### 1. 项目简介

本项目参加 Kaggle 竞赛 **PlantCLEF 2026**，目标是从样方（quadrat）照片中预测存在的植物物种。每张测试图片是一个 50×50cm 的地面植被样方照片，需要识别其中所有可见的植物物种（多标签分类）。

### 2. 技术方案

| 组件 | 选择 | 说明 |
|------|------|------|
| 模型 | ConvNeXt-Base | ImageNet-1K 预训练，1024维特征 |
| 分类头 | Linear(1024, 7806) | 7806个物种的多标签输出 |
| 损失函数 | Asymmetric Loss (ASL) | γ_neg=4, γ_pos=1, clip=0.05 |
| 训练优化 | AMP 混合精度 | 适配 RTX 3050 8G 显存 |
| 推理策略 | sigmoid + 阈值截断 | THRESHOLD=0.15, fallback=top1 |

### 3. 评分公式

$$Score = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i} \sum_{j=1}^{T_i} F1^j$$

其中 $F1^j = \frac{2 \cdot Precision_j \cdot Recall_j}{Precision_j + Recall_j}$

### 4. 文件说明

| 文件 | 说明 |
|------|------|
| `main.py` | 训练脚本：数据加载 + ConvNeXt-Base + ASL + AMP训练 |
| `predict.py` | 预测脚本：加载权重 + sigmoid推理 + 阈值截断 + 导出CSV |
| `test.py` | GPU 环境检测脚本 |
| `best_model.pth` | 训练产出的最佳模型权重 |
| `submission.csv` | Kaggle 提交文件 |

### 5. 运行方式

```bash
# 激活虚拟环境
.venv\Scripts\activate

# 训练模型 (需要训练图片在 data/train/ 目录)
python main.py

# 生成预测提交文件
python predict.py
```

### 6. 提交格式

```csv
"quadrat_id","species_ids"
"CBN-Pla-B1-20130724","[1395806]"
"CBN-PdlC-A1-20130807","[1351284, 1494911, 1381367, 1396535]"
```

### 7. 数据集结构

```
data/
├── PlantCLEF2024_single_plant_training_metadata.csv  # 训练元数据 (140万条)
├── PlantCLEF2025_test.csv                            # 测试集列表 (2105个样方)
├── species_ids.csv                                    # 有效物种ID (7806个)
├── train/                                             # 训练图片 (需下载)
│   ├── {species_id}/
│   │   └── {image_name}.jpg
└── PlantCLEF2025_test_images/
    └── PlantCLEF2025_test_images/                    # 测试样方照片 (2105张)
```
