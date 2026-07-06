# -*- coding: utf-8 -*-
"""
子群体偏差分析：按类别拆分的性能差异评估、算法偏见检测、公平性可视化

对应任务书 CO4 要求：
  - 分析模型在不同子群体上的表现差异
  - 评估潜在的算法偏见
  - 体现"技术向善"意识
"""
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import config

# 中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def get_predictions(model, dataloader, device):
    """获取模型在数据集上的所有预测结果（与 evaluate.py 中一致）"""
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='子群体偏差评估中'):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            _, preds = outputs.max(1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def compute_per_class_metrics(labels, preds, probs=None, class_names=None):
    """
    计算每个类别的详细性能指标，用于子群体偏差分析

    Args:
        labels: 真实标签 (N,)
        preds: 预测标签 (N,)
        probs: 预测概率 (N, C)，用于计算 AUC
        class_names: 类别名称列表

    Returns:
        per_class: 每个类别的指标字典列表
        disparity: 各类别间最大差异摘要
    """
    if class_names is None:
        class_names = config.CLASS_NAMES

    num_classes = len(class_names)
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    per_class = []
    for i in range(num_classes):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp

        # 类别级指标
        precision_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_i    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_i        = 2 * precision_i * recall_i / (precision_i + recall_i) if (precision_i + recall_i) > 0 else 0.0
        specificity_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        fpr_i = fp / (fp + tn) if (fp + tn) > 0 else 0.0   # 假正率
        fnr_i = fn / (fn + tp) if (fn + tp) > 0 else 0.0   # 假负率
        support_i = int(cm[i, :].sum())

        # 最常见的误判目标（该类别被误判为哪个类别最多）
        confusion_row = cm[i, :].copy()
        confusion_row[i] = 0  # 排除自身
        most_confused_with = class_names[int(np.argmax(confusion_row))] if confusion_row.sum() > 0 else 'N/A'
        confusion_pct = confusion_row.max() / cm[i, :].sum() * 100 if cm[i, :].sum() > 0 else 0.0

        per_class.append({
            'class':           class_names[i],
            'display_name':    config.CLASS_DISPLAY.get(class_names[i], class_names[i]),
            'support':         support_i,
            'precision':       precision_i,
            'recall':          recall_i,
            'f1':              f1_i,
            'specificity':     specificity_i,
            'fpr':             fpr_i,
            'fnr':             fnr_i,
            'most_confused':   most_confused_with,
            'confusion_pct':   confusion_pct,
        })

    # 计算各类别间最大差异（公平性指标）
    precisions = [c['precision'] for c in per_class]
    recalls    = [c['recall']    for c in per_class]
    f1s        = [c['f1']        for c in per_class]
    fprs       = [c['fpr']       for c in per_class]
    fnrs       = [c['fnr']       for c in per_class]

    disparity = {
        'precision_range': max(precisions) - min(precisions),
        'recall_range':    max(recalls)    - min(recalls),
        'f1_range':        max(f1s)        - min(f1s),
        'fpr_range':       max(fprs)       - min(fprs),
        'fnr_range':       max(fnrs)       - min(fnrs),
        'best_class':      per_class[np.argmax(f1s)]['display_name'],
        'worst_class':     per_class[np.argmin(f1s)]['display_name'],
        'best_f1':         max(f1s),
        'worst_f1':        min(f1s),
    }

    return per_class, disparity


def print_per_class_report(per_class, disparity):
    """打印每个类别的详细偏差分析报告"""
    print('\n' + '=' * 80)
    print('  子群体偏差分析 — 各类别性能报告')
    print('=' * 80)
    header = (f'{"类别":<14} {"样本数":>6} {"精确率":>8} {"召回率":>8} '
              f'{"F1":>8} {"特异度":>8} {"FPR":>8} {"FNR":>8} {"最易混淆":<10}')
    print(header)
    print('-' * 80)

    for c in per_class:
        print(f'{c["display_name"]:<14} {c["support"]:>6} {c["precision"]:>8.4f} '
              f'{c["recall"]:>8.4f} {c["f1"]:>8.4f} {c["specificity"]:>8.4f} '
              f'{c["fpr"]:>8.4f} {c["fnr"]:>8.4f} '
              f'{config.CLASS_DISPLAY.get(c["most_confused"], c["most_confused"]):<10}')

    print('-' * 80)
    print(f'\n公平性差异摘要:')
    print(f'  精确率差异范围:  {disparity["precision_range"]:.4f}')
    print(f'  召回率差异范围:  {disparity["recall_range"]:.4f}')
    print(f'  F1分数差异范围:  {disparity["f1_range"]:.4f}')
    print(f'  假正率差异范围:  {disparity["fpr_range"]:.4f}')
    print(f'  假负率差异范围:  {disparity["fnr_range"]:.4f}')
    print(f'  最佳类别: {disparity["best_class"]} (F1={disparity["best_f1"]:.4f})')
    print(f'  最差类别: {disparity["worst_class"]} (F1={disparity["worst_f1"]:.4f})')
    print('=' * 80)

    return per_class, disparity


def plot_per_class_metrics(per_class, save_path=None):
    """
    绘制各类别性能雷达图 + 指标柱状对比图
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ---- 左图：各类别指标柱状图 ----
    names = [c['display_name'] for c in per_class]
    x = np.arange(len(names))
    width = 0.2

    metrics_data = {
        'Precision': [c['precision'] for c in per_class],
        'Recall':    [c['recall']    for c in per_class],
        'F1':        [c['f1']        for c in per_class],
    }
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    for idx, (label, values) in enumerate(metrics_data.items()):
        bars = axes[0].bar(x + idx * width - width, values, width, label=label, color=colors[idx], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            axes[0].annotate(f'{h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                             xytext=(0, 2), textcoords='offset points', ha='center', fontsize=7)

    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=10)
    axes[0].set_ylabel('分数')
    axes[0].set_title('各类别性能指标对比', fontsize=12)
    axes[0].legend(loc='lower right')
    axes[0].set_ylim(0, 1.05)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ---- 右图：FPR vs FNR 对比（越低越好，越低越公平） ----
    fprs = [c['fpr'] for c in per_class]
    fnrs = [c['fnr'] for c in per_class]

    axes[1].bar(x - 0.15, fprs, 0.3, label='假正率 (FPR)', color='#E53935', alpha=0.85)
    axes[1].bar(x + 0.15, fnrs, 0.3, label='假负率 (FNR)', color='#FF7043', alpha=0.85)

    for i in range(len(names)):
        axes[1].text(i - 0.15, fprs[i] + 0.01, f'{fprs[i]:.3f}', ha='center', fontsize=7)
        axes[1].text(i + 0.15, fnrs[i] + 0.01, f'{fnrs[i]:.3f}', ha='center', fontsize=7)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, fontsize=10)
    axes[1].set_ylabel('错误率')
    axes[1].set_title('各类别假正率(FPR) vs 假负率(FNR)', fontsize=12)
    axes[1].legend()
    axes[1].set_ylim(0, max(max(fprs), max(fnrs)) * 1.3 + 0.05)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'fairness_per_class.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'各类别偏差分析图已保存至: {save_path}')


def plot_confusion_flow(labels, preds, class_names=None, save_path=None):
    """
    绘制混淆流向图（热力图形式），突出显示类别间误判模式
    """
    if class_names is None:
        class_names = [config.CLASS_DISPLAY.get(c, c) for c in config.CLASS_NAMES]

    cm = confusion_matrix(labels, preds)

    fig, ax = plt.subplots(figsize=(10, 8))

    # 使用对数刻度使小值可见，突出误判模式
    cm_log = cm.copy().astype(float)
    cm_log[cm_log == 0] = 1  # 避免 log(0)

    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax, linewidths=1, linecolor='white',
                cbar_kws={'label': '样本数'})

    ax.set_xlabel('预测标签', fontsize=12)
    ax.set_ylabel('真实标签', fontsize=12)
    ax.set_title('混淆矩阵 — 类别间误判模式分析', fontsize=14)

    # 高亮非对角线元素（误判）
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i != j and cm[i, j] > 0:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                           edgecolor='red', lw=2.5, linestyle='--'))

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'confusion_flow.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'混淆流向图已保存至: {save_path}')


def plot_fairness_disparity(per_class, disparity, save_path=None):
    """
    绘制公平性差异图：每个类别的 F1 与均值的偏差
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [c['display_name'] for c in per_class]
    f1_values = [c['f1'] for c in per_class]
    support_values = [c['support'] for c in per_class]
    mean_f1 = np.mean(f1_values)

    # ---- 左图：F1 偏差棒棒糖图 ----
    deviations = [v - mean_f1 for v in f1_values]
    colors_dev = ['#4CAF50' if d >= 0 else '#F44336' for d in deviations]

    axes[0].hlines(y=range(len(names)), xmin=0, xmax=deviations, colors=colors_dev, lw=3)
    axes[0].scatter(deviations, range(len(names)), c=colors_dev, s=150, zorder=5, edgecolors='white', lw=1)
    axes[0].axvline(x=0, color='gray', linestyle='--', lw=1, alpha=0.5)

    for i, d in enumerate(deviations):
        sign = '+' if d >= 0 else ''
        axes[0].text(d + (0.005 if d >= 0 else -0.005), i,
                     f'{sign}{d:.4f}', va='center',
                     ha='left' if d >= 0 else 'right', fontsize=9)

    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names)
    axes[0].set_xlabel(f'F1 偏差 (均值 = {mean_f1:.4f})')
    axes[0].set_title('各类别 F1 与均值偏差', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='x')

    # ---- 右图：样本数与F1的关系（检查是否为数据量导致偏差） ----
    axes[1].scatter(support_values, f1_values, s=200, c='#2196F3', alpha=0.7, edgecolors='white', lw=1)
    for i, name in enumerate(names):
        axes[1].annotate(name, (support_values[i], f1_values[i]),
                         xytext=(5, 5), textcoords='offset points', fontsize=9)

    # 拟合趋势线
    if len(support_values) > 1:
        z = np.polyfit(support_values, f1_values, 1)
        p = np.poly1d(z)
        x_range = np.linspace(min(support_values) * 0.9, max(support_values) * 1.1, 100)
        axes[1].plot(x_range, p(x_range), 'r--', lw=1, alpha=0.5,
                     label=f'趋势 (斜率={z[0]:.6f})')

    axes[1].set_xlabel('训练样本数')
    axes[1].set_ylabel('F1 分数')
    axes[1].set_title('样本数与分类性能的关系', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'fairness_disparity.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'公平性差异图已保存至: {save_path}')


def run_fairness_analysis(model, test_loader, device, model_name='model'):
    """
    运行完整的子群体偏差分析

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        model_name: 模型名称（用于文件命名）

    Returns:
        per_class: 各类别详细指标
        disparity: 公平性差异摘要
    """
    print('\n' + '#' * 70)
    print('#  子群体偏差与算法公平性分析')
    print('#' * 70)

    # 获取预测
    labels, preds, probs = get_predictions(model, test_loader, device)

    # 各类别指标
    per_class, disparity = compute_per_class_metrics(labels, preds, probs)

    # 打印报告
    print_per_class_report(per_class, disparity)

    # 生成可视化
    prefix = f'{model_name}_' if model_name else ''

    plot_per_class_metrics(
        per_class,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}fairness_per_class.png')
    )

    plot_confusion_flow(
        labels, preds,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}confusion_flow.png')
    )

    plot_fairness_disparity(
        per_class, disparity,
        save_path=os.path.join(config.FIGURE_DIR, f'{prefix}fairness_disparity.png')
    )

    # 输出偏差分析结论
    print('\n--- 偏差分析结论 ---')
    print(f'各类别间 F1 最大差异: {disparity["f1_range"]:.4f}')
    if disparity['f1_range'] > 0.15:
        print('⚠ 警告：类别间性能差异较大（>15%），存在明显的算法偏差风险。')
        print(f'  最差类别 "{disparity["worst_class"]}" 可能需要更多训练数据或针对性优化。')
    else:
        print('✓ 类别间性能差异在可接受范围内（<15%）。')

    # 分析是否与数据量相关
    supports = [c['support'] for c in per_class]
    f1s = [c['f1'] for c in per_class]
    if len(supports) > 1:
        corr = np.corrcoef(supports, f1s)[0, 1]
        if abs(corr) > 0.5:
            print(f'  样本量与F1的相关系数为 {corr:.3f}，说明类别不平衡是导致偏差的重要因素。')
        else:
            print(f'  样本量与F1的相关系数为 {corr:.3f}，偏差可能来源于类别本身的识别难度差异。')

    # 医疗诊断特别关注：假负率（漏诊风险）
    print('\n--- 临床风险评估（按假负率 FNR 排序） ---')
    sorted_by_fnr = sorted(per_class, key=lambda x: x['fnr'], reverse=True)
    for c in sorted_by_fnr:
        risk_level = '🔴 高' if c['fnr'] > 0.3 else ('🟡 中' if c['fnr'] > 0.15 else '🟢 低')
        print(f'  {risk_level} {c["display_name"]}: FNR={c["fnr"]:.4f} (漏诊率={c["fnr"]*100:.1f}%)')

    print('#' * 70)

    return per_class, disparity


if __name__ == '__main__':
    from dataset import create_dataloaders
    from models import get_model
    from utils import set_seed, get_device

    set_seed(config.SEED)
    device = get_device()

    _, _, test_loader = create_dataloaders()
    model = get_model('resnet50_cbam').to(device)

    # 尝试加载最佳模型
    best_path = os.path.join(config.MODEL_DIR, 'ResNet_best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('已加载最佳模型')

    run_fairness_analysis(model, test_loader, device, model_name='resnet50_cbam')
