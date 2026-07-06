# -*- coding: utf-8 -*-
"""
训练启动脚本：将输出同时写入日志文件和控制台
"""
import sys
import os
import io

# 解决Windows下控制台编码问题
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

class Tee:
    """同时输出到控制台和文件"""
    def __init__(self, *files):
        self.files = files
    def write(self, data):
        for f in self.files:
            try:
                f.write(data)
                f.flush()
            except:
                pass
    def flush(self):
        for f in self.files:
            try:
                f.flush()
            except:
                pass

# 设置日志文件
log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'training_output.txt')
log_file = open(log_path, 'w', encoding='utf-8')
sys.stdout = Tee(sys.stdout, log_file)
sys.stderr = Tee(sys.stderr, log_file)

print(f'日志文件: {log_path}')
print(f'Python: {sys.version}')
print(f'工作目录: {os.getcwd()}')

import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')

# 导入项目模块
import config
from utils import set_seed, count_parameters
from dataset import create_dataloaders
from models import get_model
from train import train
from evaluate import evaluate_model

# 解析参数
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='resnet50_cbam')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--no_aug', action='store_true')
args = parser.parse_args()

print(f'\n{"="*70}')
print(f'  训练配置')
print(f'{"="*70}')
print(f'  模型: {args.model}')
print(f'  轮数: {args.epochs}')
print(f'  数据增强: {"否" if args.no_aug else "是"}')
print(f'  图像尺寸: {config.IMG_SIZE}')
print(f'  批大小: {config.BATCH_SIZE}')
print(f'  学习率: {config.LR}')

set_seed(config.SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建数据加载器
print('\n加载数据集...')
train_loader, val_loader, test_loader = create_dataloaders(use_augmentation=not args.no_aug)
print(f'训练集: {len(train_loader.dataset)} 样本')
print(f'验证集: {len(val_loader.dataset)} 样本')
print(f'测试集: {len(test_loader.dataset)} 样本')

# 创建模型
print(f'\n创建模型: {args.model}...')
model = get_model(args.model)
num_params = count_parameters(model)
print(f'模型参数量: {num_params:,}')

# 训练
print('\n开始训练...')
history = train(model, train_loader, val_loader)

# 加载最佳模型
best_model_path = os.path.join(config.MODEL_DIR, f'{model.__class__.__name__}_best.pth')
if os.path.exists(best_model_path):
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'\n已加载最佳模型: {best_model_path}')

# 评估
print(f'\n评估模型...')
metrics = evaluate_model(model, test_loader, device=device, history=history, model_name=args.model)

print(f'\n{"="*70}')
print(f'  训练完成！')
print(f'{"="*70}')
print(f'  测试准确率: {metrics["accuracy"]*100:.2f}%')
print(f'  F1 (macro): {metrics["f1_macro"]:.4f}')
print(f'  F1 (weighted): {metrics["f1_weighted"]:.4f}')

log_file.close()
