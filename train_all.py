# -*- coding: utf-8 -*-
"""
一键训练脚本：依次训练所有模型并运行消融实验
"""
import os
import sys
import time
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils import set_seed, count_parameters
from dataset import create_dataloaders
from models import get_model
from train import train
from evaluate import evaluate_model


def train_single_model(model_name, epochs=30, no_aug=False):
    """训练单个模型"""
    print(f'\n{"="*70}')
    print(f'  开始训练: {model_name}')
    print(f'{"="*70}')

    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备: {device}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name(0)}')
        print(f'显存: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(use_augmentation=not no_aug)

    # 创建模型
    model = get_model(model_name)
    num_params = count_parameters(model)
    print(f'模型参数量: {num_params:,}')

    # 训练
    history = train(model, train_loader, val_loader)

    # 加载最佳模型进行评估
    best_model_path = os.path.join(config.MODEL_DIR, f'{model.__class__.__name__}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'已加载最佳模型: {best_model_path}')

    # 评估
    print(f'\n评估模型 {model_name} 在测试集上的表现...')
    metrics = evaluate_model(model, test_loader, device=device, history=history, model_name=model_name)

    return history, metrics


def run_ablation(epochs=30):
    """运行消融实验"""
    print(f'\n{"="*70}')
    print(f'  消融实验')
    print(f'{"="*70}')

    ablation_configs = [
        ('baseline',       True,  'Baseline CNN（无增强、无注意力）'),
        ('resnet50',       True,  'ResNet50（无增强、无注意力）'),
        ('resnet50',       False, 'ResNet50 + 数据增强'),
        ('resnet50_cbam',  True,  'ResNet50 + CBAM（无增强）'),
        ('resnet50_cbam',  False, 'ResNet50 + CBAM + 数据增强（完整方案）'),
    ]

    results = []
    for model_name, no_aug, desc in ablation_configs:
        print(f'\n--- 消融实验: {desc} ---')
        history, metrics = train_single_model(model_name, epochs=epochs, no_aug=no_aug)
        results.append({
            'description': desc,
            'model_name': model_name,
            'no_aug': no_aug,
            'metrics': metrics,
        })

    # 打印消融实验汇总
    print(f'\n{"="*70}')
    print(f'  消融实验汇总')
    print(f'{"="*70}')
    print(f'{"配置":<35} {"准确率":>8} {"F1(macro)":>10} {"F1(weighted)":>13}')
    print('-' * 65)
    for r in results:
        m = r['metrics']
        print(f'{r["description"]:<35} {m["accuracy"]:>7.4f} {m["f1_macro"]:>10.4f} {m["f1_weighted"]:>13.4f}')

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='脑肿瘤分类训练脚本')
    parser.add_argument('--model', type=str, default='resnet50_cbam',
                        choices=config.MODEL_NAMES, help='模型名称')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--ablation', action='store_true', help='运行消融实验')
    parser.add_argument('--no_aug', action='store_true', help='不使用数据增强')
    args = parser.parse_args()

    if args.ablation:
        run_ablation(epochs=args.epochs)
    else:
        train_single_model(args.model, epochs=args.epochs, no_aug=args.no_aug)
