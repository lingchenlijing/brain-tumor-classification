# -*- coding: utf-8 -*-
"""
配置文件：集中管理所有超参数、路径和模型名称
"""
import os

# ======================== 数据集路径 ========================
# 数据集根目录
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'Brain-Tumor-Classification-DataSet-master')
TRAIN_DIR = os.path.join(DATA_ROOT, 'Training')
TEST_DIR = os.path.join(DATA_ROOT, 'Testing')

# ======================== 输出目录 ========================
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')

# 确保输出目录存在
for d in [OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================== 超参数 ========================
IMG_SIZE = 224           # 输入图像尺寸
BATCH_SIZE = 16          # 批大小（RTX 3060 Laptop 6GB显存，ResNet50需降低batch_size）
NUM_EPOCHS = 30          # 训练轮数
LR = 0.001               # 初始学习率
WEIGHT_DECAY = 1e-4      # 权重衰减
LABEL_SMOOTHING = 0.1    # 标签平滑
NUM_CLASSES = 4          # 分类数
VAL_RATIO = 0.15         # 验证集比例
SEED = 42                # 随机种子
EARLY_STOP_PATIENCE = 7  # 早停耐心值
NUM_WORKERS = 0          # DataLoader 工作线程数（Windows下设为0避免多进程问题）

# ======================== 类别信息 ========================
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

# 类别中文显示名称（用于图表和报告）
CLASS_DISPLAY = {
    'glioma_tumor':      '胶质瘤',
    'meningioma_tumor':  '脑膜瘤',
    'no_tumor':          '无肿瘤',
    'pituitary_tumor':   '垂体瘤',
}
CLASS_DISPLAY_CN = ['胶质瘤', '脑膜瘤', '无肿瘤', '垂体瘤']  # 与 CLASS_NAMES 一一对应

# ======================== 模型名称 ========================
MODEL_NAMES = ['baseline', 'resnet18', 'resnet50', 'resnet50_cbam']

# ======================== ImageNet 归一化参数 ========================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
