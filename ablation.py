# -*- coding: utf-8 -*-
"""
消融实验：逐步验证数据增强和 CBAM 注意力机制的贡献
实验配置：
  1. Baseline CNN（无增强、无注意力）
  2. ResNet50（无增强、无注意力）
  3. ResNet50 + 数据增强
  4. ResNet50 + CBAM（无增强）
  5. ResNet50 + 数据增强 + CBAM（完整模型）
"""
import os
import copy
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import config
from dataset import create_dataloaders
from models import get_model
from train import train
from evaluate import evaluate_model, compute_metrics, get_predictions
from utils import set_seed, get_device, count_parameters


# 消融实验配置
ABLATION_CONFIGS = [
    {
        'name': 'Baseline CNN',
        'model_name': 'baseline',
        'use_augmentation': False,
        'label': 'Baseline CNN',
    },
    {
        'name': 'ResNet50',
        'model_name': 'resnet50',
        'use_augmentation': False,
        'label': 'ResNet50',
    },
    {
        'name': 'ResNet50 + Aug',
        'model_name': 'resnet50',
        'use_augmentation': True,
        'label': 'ResNet50 + 数据增强',
    },
    {
        'name': 'ResNet50 + CBAM',
        'model_name': 'resnet50_cbam',
        'use_augmentation': False,
        'label': 'ResNet50 + CBAM',
    },
    {
        'name': 'ResNet50 + Aug + CBAM',
        'model_name': 'resnet50_cbam',
        'use_augmentation': True,
        'label': 'ResNet50 + 数据增强 + CBAM',
    },
]


def run_single_experiment(exp_config, num_epochs=None):
    """
    运行单个消融实验

    Args:
        exp_config: 实验配置字典
        num_epochs: 训练轮数（默认使用 config.NUM_EPOCHS）

    Returns:
        results: 包含训练历史和评估指标的字典
    """
    set_seed(config.SEED)
    device = get_device()

    # 临时修改训练轮数
    original_epochs = config.NUM_EPOCHS
    if num_epochs is not None:
        config.NUM_EPOCHS = num_epochs

    print(f'\n{"=" * 60}')
    print(f'实验: {exp_config["label"]}')
    print(f'模型: {exp_config["model_name"]}')
    print(f'数据增强: {exp_config["use_augmentation"]}')
    print(f'{"=" * 60}')

    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        use_augmentation=exp_config['use_augmentation']
    )

    # 创建模型
    model = get_model(exp_config['model_name'])
    print(f'模型参数量: {count_parameters(model):,}')

    # 临时修改检查点命名，避免不同实验覆盖彼此的检查点
    original_model_name = model.__class__.__name__
    unique_model_name = f"{original_model_name}_ablation_{exp_config['name'].replace(' ', '_').replace('+', 'plus')}"
    model.__class__.__name__ = unique_model_name

    # 训练
    history = train(model, train_loader, val_loader)

    # 加载最佳模型进行评估
    best_model_path = os.path.join(config.MODEL_DIR, f'{unique_model_name}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    # 在测试集上评估
    labels, preds, probs = get_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, preds, probs)

    # 恢复原始训练轮数和模型名称
    config.NUM_EPOCHS = original_epochs
    model.__class__.__name__ = original_model_name

    return {
        'name': exp_config['name'],
        'label': exp_config['label'],
        'history': history,
        'metrics': metrics,
        'params': count_parameters(model),
    }


def run_ablation_study(num_epochs=None):
    """
    运行完整的消融实验

    Args:
        num_epochs: 训练轮数（默认使用 config.NUM_EPOCHS）

    Returns:
        all_results: 所有实验结果的列表
    """
    all_results = []

    for i, exp_config in enumerate(ABLATION_CONFIGS):
        print(f'\n>>> 消融实验 [{i + 1}/{len(ABLATION_CONFIGS)}]: {exp_config["label"]}')
        result = run_single_experiment(exp_config, num_epochs)
        all_results.append(result)

    # 打印汇总表
    print_summary_table(all_results)

    # 绘制对比图
    plot_ablation_comparison(all_results)

    return all_results


def print_summary_table(results):
    """打印消融实验汇总表"""
    print(f'\n{"=" * 90}')
    print('消融实验汇总表')
    print(f'{"=" * 90}')
    header = f'{"实验配置":<28} {"参数量":>10} {"准确率":>8} {"F1(macro)":>10} {"F1(weighted)":>13} {"AUC(macro)":>10}'
    print(header)
    print('-' * 90)

    for r in results:
        m = r['metrics']
        auc_str = f'{m["auc_macro"]:.4f}' if 'auc_macro' in m else 'N/A'
        print(f'{r["label"]:<28} {r["params"]:>10,} {m["accuracy"]:>8.4f} '
              f'{m["f1_macro"]:>10.4f} {m["f1_weighted"]:>13.4f} {auc_str:>10}')

    print(f'{"=" * 90}')


def plot_ablation_comparison(results, save_path=None):
    """
    绘制消融实验对比柱状图

    Args:
        results: 实验结果列表
        save_path: 保存路径
    """
    labels = [r['label'] for r in results]
    accuracies = [r['metrics']['accuracy'] * 100 for r in results]
    f1_macros = [r['metrics']['f1_macro'] * 100 for r in results]
    f1_weighteds = [r['metrics']['f1_weighted'] * 100 for r in results]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width, accuracies, width, label='准确率', color='#2196F3', alpha=0.85)
    bars2 = ax.bar(x, f1_macros, width, label='F1 (macro)', color='#FF9800', alpha=0.85)
    bars3 = ax.bar(x + width, f1_weighteds, width, label='F1 (weighted)', color='#4CAF50', alpha=0.85)

    # 在柱子上方标注数值
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords='offset points',
                        ha='center', va='bottom', fontsize=8)

    ax.set_xlabel('实验配置')
    ax.set_ylabel('指标值 (%)')
    ax.set_title('消融实验对比')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.legend()
    ax.set_ylim(0, max(max(accuracies), max(f1_macros), max(f1_weighteds)) + 10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'ablation_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'消融实验对比图已保存至: {save_path}')


if __name__ == '__main__':
    # 运行消融实验
    results = run_ablation_study()
