# -*- coding: utf-8 -*-
"""
评估与可视化：分类报告、混淆矩阵、ROC 曲线、训练曲线、预测示例
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms as T
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import config

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def _to_display_names(class_names):
    """将英文类别名转换为中文显示名，用于图表标签"""
    if class_names is None:
        return config.CLASS_DISPLAY_CN
    return [config.CLASS_DISPLAY.get(c, c) for c in class_names]


def get_predictions(model, dataloader, device):
    """
    获取模型在数据集上的所有预测结果

    Returns:
        all_labels: 真实标签 (numpy array)
        all_preds: 预测标签 (numpy array)
        all_probs: 预测概率 (numpy array, shape: [N, num_classes])
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='评估中'):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_metrics(labels, preds, probs=None):
    """
    计算各种评估指标

    Returns:
        metrics: 包含各项指标的字典
    """
    metrics = {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'precision_weighted': precision_score(labels, preds, average='weighted', zero_division=0),
        'recall_weighted': recall_score(labels, preds, average='weighted', zero_division=0),
        'f1_weighted': f1_score(labels, preds, average='weighted', zero_division=0),
    }

    # 计算 AUC（如果有概率输出）
    if probs is not None:
        labels_bin = label_binarize(labels, classes=list(range(config.NUM_CLASSES)))
        # 计算每个类别的 AUC
        auc_values = []
        for i in range(config.NUM_CLASSES):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            auc_values.append(auc(fpr, tpr))
        metrics['auc_macro'] = np.mean(auc_values)

    return metrics


def print_classification_report(labels, preds, class_names=None):
    """打印分类报告"""
    if class_names is None:
        class_names = config.CLASS_NAMES
    display_names = _to_display_names(class_names)
    report = classification_report(labels, preds, target_names=display_names, digits=4)
    print('\n分类报告:')
    print(report)
    return report


def plot_confusion_matrix(labels, preds, class_names=None, save_path=None):
    """
    绘制混淆矩阵

    Args:
        labels: 真实标签
        preds: 预测标签
        class_names: 类别名称列表
        save_path: 保存路径
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    display_names = _to_display_names(class_names)

    cm = confusion_matrix(labels, preds)
    # 归一化
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 原始计数
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=display_names,
                yticklabels=display_names, ax=axes[0])
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')
    axes[0].set_title('混淆矩阵 (计数)')

    # 归一化比例
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues', xticklabels=display_names,
                yticklabels=display_names, ax=axes[1])
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')
    axes[1].set_title('混淆矩阵 (比例)')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'混淆矩阵已保存至: {save_path}')


def plot_roc_curves(labels, probs, class_names=None, save_path=None):
    """
    绘制 ROC 曲线（One-vs-Rest）

    Args:
        labels: 真实标签
        probs: 预测概率 (N, num_classes)
        class_names: 类别名称列表
        save_path: 保存路径
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    display_names = _to_display_names(class_names)

    # 二值化标签
    labels_bin = label_binarize(labels, classes=list(range(config.NUM_CLASSES)))

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, color in zip(range(config.NUM_CLASSES), colors):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{display_names[i]} (AUC = {roc_auc:.4f})')

    # 微平均 ROC
    fpr_micro, tpr_micro, _ = roc_curve(labels_bin.ravel(), probs.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)
    ax.plot(fpr_micro, tpr_micro, color='navy', lw=2, linestyle=':',
            label=f'微平均 (AUC = {roc_auc_micro:.4f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('假正率 (FPR)')
    ax.set_ylabel('真正率 (TPR)')
    ax.set_title('ROC 曲线 (One-vs-Rest)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'roc_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'ROC 曲线已保存至: {save_path}')


def plot_training_curves(history, save_path=None):
    """
    绘制训练曲线（损失和准确率）

    Args:
        history: 训练历史字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # 损失曲线
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练损失', lw=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='验证损失', lw=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('损失')
    axes[0].set_title('训练/验证损失曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', lw=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', lw=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('准确率 (%)')
    axes[1].set_title('训练/验证准确率曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'训练曲线已保存至: {save_path}')


def plot_prediction_examples(model, dataloader, device, class_names=None,
                             num_examples=8, save_path=None):
    """
    展示正确和错误预测的示例

    Args:
        model: 模型
        dataloader: 数据加载器
        device: 计算设备
        class_names: 类别名称列表
        num_examples: 每类展示的示例数
        save_path: 保存路径
    """
    if class_names is None:
        class_names = config.CLASS_NAMES
    display_names = _to_display_names(class_names)

    model.eval()
    correct_images = []
    correct_labels = []
    correct_preds = []
    incorrect_images = []
    incorrect_labels = []
    incorrect_preds = []

    # 反归一化
    inv_normalize = T.Normalize(
        mean=[-m / s for m, s in zip(config.IMAGENET_MEAN, config.IMAGENET_STD)],
        std=[1.0 / s for s in config.IMAGENET_STD]
    )

    with torch.no_grad():
        for images, labels in dataloader:
            images_dev = images.to(device)
            outputs = model(images_dev)
            _, preds = outputs.max(1)
            preds = preds.cpu()

            for i in range(images.size(0)):
                img = inv_normalize(images[i]).clamp(0, 1)
                if preds[i] == labels[i]:
                    if len(correct_images) < num_examples:
                        correct_images.append(img)
                        correct_labels.append(labels[i].item())
                        correct_preds.append(preds[i].item())
                else:
                    if len(incorrect_images) < num_examples:
                        incorrect_images.append(img)
                        incorrect_labels.append(labels[i].item())
                        incorrect_preds.append(preds[i].item())

            if len(correct_images) >= num_examples and len(incorrect_images) >= num_examples:
                break

    # 绘制正确预测
    n_correct = len(correct_images)
    n_incorrect = len(incorrect_images)
    n_total = n_correct + n_incorrect

    if n_total == 0:
        print('没有找到预测示例')
        return

    fig, axes = plt.subplots(2, max(n_correct, n_incorrect), figsize=(3 * max(n_correct, n_incorrect), 6))
    if max(n_correct, n_incorrect) == 1:
        axes = axes.reshape(2, 1)

    # 正确预测行
    for i in range(max(n_correct, n_incorrect)):
        if i < n_correct:
            img = correct_images[i].permute(1, 2, 0).numpy()
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'✓ {display_names[correct_preds[i]]}', fontsize=8, color='green')
        axes[0, i].axis('off')

    # 错误预测行
    for i in range(max(n_correct, n_incorrect)):
        if i < n_incorrect:
            img = incorrect_images[i].permute(1, 2, 0).numpy()
            axes[1, i].imshow(img)
            axes[1, i].set_title(
                f'✗ 真:{display_names[incorrect_labels[i]]}\n预:{display_names[incorrect_preds[i]]}',
                fontsize=7, color='red'
            )
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('正确预测', fontsize=10)
    axes[1, 0].set_ylabel('错误预测', fontsize=10)

    plt.suptitle('预测示例', fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'prediction_examples.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'预测示例已保存至: {save_path}')


def evaluate_model(model, test_loader, device, history=None, model_name='model',
                   run_fairness=True):
    """
    完整评估流程：计算指标、打印报告、生成所有可视化、子群体偏差分析

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        history: 训练历史（用于绘制训练曲线）
        model_name: 模型名称（用于文件命名）
        run_fairness: 是否运行子群体偏差分析（默认 True，对应 CO4 要求）

    Returns:
        metrics: 评估指标字典
    """
    print('\n' + '=' * 60)
    print('开始模型评估')
    print('=' * 60)

    # 获取预测结果
    labels, preds, probs = get_predictions(model, test_loader, device)

    # 计算指标
    metrics = compute_metrics(labels, preds, probs)

    # 打印结果
    print(f'\n测试集评估结果:')
    print(f'  准确率:           {metrics["accuracy"]:.4f}')
    print(f'  精确率 (macro):   {metrics["precision_macro"]:.4f}')
    print(f'  召回率 (macro):   {metrics["recall_macro"]:.4f}')
    print(f'  F1 (macro):       {metrics["f1_macro"]:.4f}')
    print(f'  精确率 (weighted):{metrics["precision_weighted"]:.4f}')
    print(f'  召回率 (weighted):{metrics["recall_weighted"]:.4f}')
    print(f'  F1 (weighted):    {metrics["f1_weighted"]:.4f}')
    if 'auc_macro' in metrics:
        print(f'  AUC (macro):      {metrics["auc_macro"]:.4f}')

    # 打印分类报告
    print_classification_report(labels, preds)

    # 生成可视化（文件名包含模型名称）
    prefix = f'{model_name}_' if model_name else ''

    plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}confusion_matrix.png')
    )

    plot_roc_curves(
        labels, probs,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}roc_curves.png')
    )

    if history is not None:
        plot_training_curves(
            history,
            save_path=os.path.join(config.FIGURE_DIR, f'{prefix}training_curves.png')
        )

    plot_prediction_examples(
        model, test_loader, device,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}prediction_examples.png')
    )

    # ====== 模型结构图（报告 §5） ======
    print('\n--- 生成模型结构图 ---')
    try:
        total_params = plot_model_architecture(
            model,
            save_path=os.path.join(config.FIGURE_DIR, f'{prefix}architecture.png')
        )
        metrics['total_params'] = total_params
    except Exception as e:
        print(f'⚠ 模型结构图生成失败: {e}')

    # ====== 子群体偏差分析（CO4 要求） ======
    if run_fairness:
        try:
            from fairness import run_fairness_analysis
            per_class, disparity = run_fairness_analysis(model, test_loader, device, model_name)
            metrics['per_class'] = per_class
            metrics['disparity'] = disparity
        except Exception as e:
            print(f'⚠ 子群体偏差分析运行失败: {e}')

    return metrics


def plot_model_architecture(model, input_size=(3, 224, 224), save_path=None):
    """
    绘制模型结构框图（用于报告 §5 模型设计与实现）

    自动解析模型层结构，用表格或框图形式展示各层的类型、
    输入/输出尺寸和参数量。

    Args:
        model: PyTorch 模型
        input_size: 输入张量形状 (C, H, W)
        save_path: 保存路径
    """
    import torch.nn as nn

    model.eval()
    device = next(model.parameters()).device
    dummy_input = torch.randn(1, *input_size, device=device)

    # 注册钩子记录每层的输入/输出形状
    layer_info = []

    def _hook(module, input, output):
        in_shape = tuple(input[0].shape)
        out_shape = tuple(output.shape)
        params = sum(p.numel() for p in module.parameters())
        layer_info.append({
            'name': module.__class__.__name__,
            'in_shape': in_shape,
            'out_shape': out_shape,
            'params': params,
        })

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.ReLU,
                          nn.MaxPool2d, nn.AdaptiveAvgPool2d, nn.Flatten,
                          nn.Dropout)):
            hooks.append(m.register_forward_hook(_hook))

    with torch.no_grad():
        _ = model(dummy_input)

    for h in hooks:
        h.remove()

    # 绘制
    n_layers = len(layer_info)
    if n_layers == 0:
        print('无法提取模型层信息')
        return

    fig, ax = plt.subplots(figsize=(16, max(8, n_layers * 0.35)))
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, n_layers + 1)
    ax.axis('off')

    # 标题
    ax.text(5, n_layers + 0.5, f'模型结构: {model.__class__.__name__}',
            ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(5, n_layers, f'输入: {input_size}   |   总层数(含激活/池化): {n_layers}',
            ha='center', va='center', fontsize=9, color='gray')

    # 颜色映射
    color_map = {
        'Conv2d': '#2196F3',
        'BatchNorm2d': '#FF9800',
        'ReLU': '#4CAF50',
        'MaxPool2d': '#F44336',
        'AdaptiveAvgPool2d': '#9C27B0',
        'Linear': '#00BCD4',
        'Flatten': '#607D8B',
        'Dropout': '#795548',
    }

    for i, info in enumerate(layer_info):
        y = n_layers - 1 - i
        color = color_map.get(info['name'], '#9E9E9E')

        # 层类型标签框
        box = plt.Rectangle((0.3, y - 0.35), 1.8, 0.7, facecolor=color, alpha=0.85,
                           edgecolor='white', lw=1.5, transform=ax.transData)
        ax.add_patch(box)
        ax.text(1.2, y, info['name'], ha='center', va='center', fontsize=8,
                color='white', fontweight='bold')

        # 输入形状
        in_str = f'({info["in_shape"][1]}, {info["in_shape"][2]}, {info["in_shape"][3]})' \
                 if len(info['in_shape']) == 4 else str(info['in_shape'])
        ax.text(2.5, y, f'输入: {in_str}', ha='left', va='center', fontsize=7, color='#555')

        # 输出形状
        out_str = f'({info["out_shape"][1]}, {info["out_shape"][2]}, {info["out_shape"][3]})' \
                  if len(info['out_shape']) == 4 else str(info['out_shape'])
        ax.text(5.0, y, f'→ 输出: {out_str}', ha='left', va='center', fontsize=7, color='#333')

        # 参数量
        if info['params'] > 0:
            if info['params'] >= 1e6:
                param_str = f'{info["params"]/1e6:.1f}M'
            elif info['params'] >= 1e3:
                param_str = f'{info["params"]/1e3:.1f}K'
            else:
                param_str = str(info['params'])
            ax.text(7.5, y, f'参数: {param_str}', ha='left', va='center', fontsize=7, color='#E91E63')

        # 连接线
        if i < n_layers - 1:
            ax.annotate('', xy=(0.5, y - 0.5), xytext=(0.5, y - 0.35),
                       arrowprops=dict(arrowstyle='->', color='gray', lw=1))

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, f'{model.__class__.__name__}_architecture.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'模型结构图已保存至: {save_path}')

    # 同时保存文本版本
    txt_path = save_path.replace('.png', '.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f'模型结构: {model.__class__.__name__}\n')
        f.write(f'输入形状: {input_size}\n')
        f.write(f'总层数: {n_layers}\n')
        f.write('=' * 70 + '\n')
        for info in layer_info:
            f.write(f'{info["name"]:<20} {str(info["in_shape"]):<25} -> {str(info["out_shape"]):<25} '
                    f'参数: {info["params"]:,}\n')
    print(f'模型结构文本已保存至: {txt_path}')

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params


if __name__ == '__main__':
    # 测试评估流程
    from dataset import create_dataloaders
    from models import get_model
    from utils import set_seed

    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    _, _, test_loader = create_dataloaders()
    model = get_model('baseline').to(device)
    metrics = evaluate_model(model, test_loader, device, model_name='baseline_test')
