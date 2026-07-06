# 脑肿瘤 MRI 图像分类

基于深度学习的脑肿瘤 MRI 图像四分类系统，使用 ResNet50-CBAM 模型对胶质瘤、脑膜瘤、垂体瘤和无肿瘤进行自动分类。

## 项目简介

本项目使用 PyTorch 构建了一个完整的医学图像分类流程，包括数据预处理、模型训练、消融实验、公平性分析、模型优化与部署导出。核心模型在标准 ResNet50 基础上融入 CBAM 注意力机制，提升模型对肿瘤区域特征的捕捉能力。

通过模块化的代码结构，可以方便地复现训练过程、对比不同模型性能，并生成完整的可视化分析图表。

## 数据集

- **名称**：Brain Tumor Classification DataSet
- **类别**：glioma_tumor、meningioma_tumor、pituitary_tumor、no_tumor
- **放置**：将 `Brain-Tumor-Classification-DataSet-master` 文件夹下载至项目根目录即可运行
- **说明**：数据集未包含在仓库中，可在 Kaggle 搜索 "Brain Tumor Classification DataSet" 获取

## 运行方式

```bash
# 1. 安装依赖
pip install torch torchvision timm scikit-learn matplotlib seaborn pandas numpy tqdm

# 2. 将 Brain-Tumor-Classification-DataSet-master 放入项目根目录

# 3. 运行完整训练流程
python run.py

# 或单独训练 ResNet50-CBAM 模型
python start_train.py

# 或运行消融实验
python ablation.py
```

## 主要功能

1. **数据预处理**
   - 图像归一化与数据增强
   - 训练集/验证集划分
   - 类别分布可视化

2. **模型训练**
   - Baseline CNN、ResNet18、ResNet50、ResNet50-CBAM 对比
   - 交叉熵损失 + 标签平滑
   - 学习率调度与早停机制
   - 训练曲线与混淆矩阵可视化

3. **消融实验**
   - CBAM 注意力模块有效性验证
   - 不同数据增强策略对比
   - 学习率与优化器组合对比

4. **公平性分析**
   - 各肿瘤类别上的性能差异
   - 混淆矩阵与 ROC 曲线

5. **模型优化与部署**
   - 模型量化与剪枝
   - ONNX / TorchScript 导出
   - 推理延迟与参数量分析

## 项目结构

```
.
├── ablation.py              # 消融实验脚本
├── config.py                # 全局配置与路径
├── dataset.py               # 数据集与 DataLoader
├── deploy.py                # 模型部署与导出
├── evaluate.py              # 模型评估指标
├── fairness.py              # 公平性分析
├── models.py                # 网络模型定义
├── optimize.py              # 模型优化（量化/剪枝）
├── run.py                   # 主运行入口
├── start_train.py           # 快速训练入口
├── train.py                 # 训练逻辑
├── train_all.py             # 多模型训练对比
├── utils.py                 # 工具函数
├── outputs/
│   └── figures/             # 生成的可视化图表
├── .gitignore
└── README.md
```

## 技术栈

- Python 3
- PyTorch / torchvision
- timm
- scikit-learn
- matplotlib / seaborn
- pandas / numpy
- tqdm

## 许可证

本项目仅用于学习与技术展示。
