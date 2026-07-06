# -*- coding: utf-8 -*-
"""
工具函数：种子设置、设备获取、参数统计、检查点保存/加载、指标追踪、早停
"""
import os
import random
import numpy as np
import torch


def set_seed(seed=42):
    """设置随机种子以保证可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 确保 CUDA 卷积操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """获取可用计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'使用 GPU: {torch.cuda.get_device_name(0)}')
    else:
        device = torch.device('cpu')
        print('使用 CPU')
    return device


def count_parameters(model):
    """统计模型可训练参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(state, filepath):
    """保存模型检查点"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(state, filepath)
    print(f'检查点已保存至: {filepath}')


def load_checkpoint(filepath, model, optimizer=None):
    """加载模型检查点，可选加载优化器状态"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f'检查点文件不存在: {filepath}')
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f'检查点已加载: {filepath}')
    return checkpoint


class AverageMeter:
    """用于追踪和计算指标的平均值"""

    def __init__(self, name='metric'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0       # 当前值
        self.avg = 0       # 平均值
        self.sum = 0       # 累计总和
        self.count = 0     # 累计计数

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0


class EarlyStopping:
    """早停机制：验证指标在指定轮数内无改善则停止训练"""

    def __init__(self, patience=7, mode='max'):
        """
        Args:
            patience: 容忍的轮数
            mode: 'max' 表示指标越大越好，'min' 表示越小越好
        """
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        # 判断是否改善
        improved = (score > self.best_score) if self.mode == 'max' else (score < self.best_score)
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                print(f'早停触发！连续 {self.patience} 轮验证指标未改善')
        return self.early_stop
