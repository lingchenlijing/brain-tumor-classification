# -*- coding: utf-8 -*-
"""
数据加载与预处理：自定义数据集类、数据增强、加权采样、DataLoader 创建
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision import datasets, transforms
import config


def get_train_transforms():
    """训练集数据增强变换"""
    return transforms.Compose([
        transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_val_transforms():
    """验证集/测试集标准变换（无增强）"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def get_no_aug_transforms():
    """无增强的变换（消融实验用）"""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
    ])


def create_dataloaders(use_augmentation=True, batch_size=None):
    """
    创建训练、验证和测试的 DataLoader

    Args:
        use_augmentation: 是否使用数据增强
        batch_size: 批大小，默认使用 config.BATCH_SIZE

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = config.BATCH_SIZE

    # 选择训练变换
    train_transform = get_train_transforms() if use_augmentation else get_no_aug_transforms()
    val_transform = get_val_transforms()

    # 加载完整训练集（用于划分训练/验证）
    full_train_dataset = datasets.ImageFolder(root=config.TRAIN_DIR, transform=train_transform)

    # 计算训练/验证划分索引
    num_total = len(full_train_dataset)
    num_val = int(num_total * config.VAL_RATIO)
    num_train = num_total - num_val

    # 使用固定种子划分，确保可复现
    generator = torch.Generator().manual_seed(config.SEED)
    indices = torch.randperm(num_total, generator=generator).tolist()
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    # 创建训练子集
    train_subset = Subset(full_train_dataset, train_indices)

    # 创建验证子集（使用验证变换，需重新创建数据集）
    val_dataset_with_transform = datasets.ImageFolder(root=config.TRAIN_DIR, transform=val_transform)
    val_subset = Subset(val_dataset_with_transform, val_indices)

    # 创建测试集
    test_dataset = datasets.ImageFolder(root=config.TEST_DIR, transform=val_transform)

    # 计算训练集类别权重用于 WeightedRandomSampler（处理类别不平衡）
    targets = [full_train_dataset.targets[i] for i in train_indices]
    class_counts = np.bincount(targets, minlength=config.NUM_CLASSES)
    # 每个样本的权重 = 其所属类别的逆频率
    sample_weights = 1.0 / class_counts[targets]
    sample_weights = torch.from_numpy(sample_weights).double()

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_indices),
        replacement=True
    )

    # 创建 DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        sampler=sampler,       # 使用加权采样器，不再 shuffle
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    # 打印数据集信息
    print(f'数据集加载完成:')
    print(f'  训练集: {len(train_subset)} 张 (含数据增强: {use_augmentation})')
    print(f'  验证集: {len(val_subset)} 张')
    print(f'  测试集: {len(test_dataset)} 张')
    print(f'  类别分布 (训练集): {dict(zip(config.CLASS_NAMES, class_counts))}')

    return train_loader, val_loader, test_loader


# ======================== EDA 可视化 ========================

def plot_class_distribution(save_path=None):
    """
    绘制训练集和测试集的类别分布柱状图

    Args:
        save_path: 保存路径，默认保存到 FIGURE_DIR/class_distribution.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    from torchvision import datasets as ds

    # 加载原始数据集（无增强）以统计
    train_dataset = ds.ImageFolder(root=config.TRAIN_DIR)
    test_dataset = ds.ImageFolder(root=config.TEST_DIR)

    train_counts = np.bincount(train_dataset.targets, minlength=config.NUM_CLASSES)
    test_counts = np.bincount(test_dataset.targets, minlength=config.NUM_CLASSES)

    display_names = config.CLASS_DISPLAY_CN

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(display_names))
    colors = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    # 训练集
    bars1 = axes[0].bar(x, train_counts, color=colors, alpha=0.85, edgecolor='white')
    for bar, count in zip(bars1, train_counts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                     f'{count}\n({count/len(train_dataset)*100:.1f}%)',
                     ha='center', fontsize=9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, fontsize=10)
    axes[0].set_ylabel('样本数', fontsize=11)
    axes[0].set_title(f'训练集类别分布 (总计 {len(train_dataset)} 张)', fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')

    # 测试集
    bars2 = axes[1].bar(x, test_counts, color=colors, alpha=0.85, edgecolor='white')
    for bar, count in zip(bars2, test_counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 3,
                     f'{count}\n({count/len(test_dataset)*100:.1f}%)',
                     ha='center', fontsize=9)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, fontsize=10)
    axes[1].set_ylabel('样本数', fontsize=11)
    axes[1].set_title(f'测试集类别分布 (总计 {len(test_dataset)} 张)', fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'class_distribution.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'类别分布图已保存至: {save_path}')

    return train_counts, test_counts


def plot_sample_images(samples_per_class=4, save_path=None):
    """
    绘制每类样本图像网格

    Args:
        samples_per_class: 每类展示的样本数
        save_path: 保存路径，默认保存到 FIGURE_DIR/sample_images.png
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    from PIL import Image

    # 扫描训练集目录获取各类别图像路径
    display_names = config.CLASS_DISPLAY_CN

    fig, axes = plt.subplots(
        config.NUM_CLASSES, samples_per_class,
        figsize=(2.5 * samples_per_class, 2.5 * config.NUM_CLASSES)
    )

    for ci, (cls_name, display_name) in enumerate(zip(config.CLASS_NAMES, display_names)):
        cls_dir = os.path.join(config.TRAIN_DIR, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        image_files = [f for f in os.listdir(cls_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        selected = image_files[:samples_per_class]

        for si, img_file in enumerate(selected):
            img_path = os.path.join(cls_dir, img_file)
            img = Image.open(img_path).convert('RGB')
            ax = axes[ci, si] if config.NUM_CLASSES > 1 else axes[si]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            if si == 0:
                ax.set_ylabel(display_name, fontsize=11, fontweight='bold')
            if ci == 0:
                ax.set_title(f'样本 {si + 1}', fontsize=9)

    plt.suptitle('各类别训练样本示例', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'sample_images.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'样本图像已保存至: {save_path}')


def run_eda(save_dir=None):
    """
    运行完整 EDA 流程，生成数据集分析所需的所有图表

    Args:
        save_dir: 保存目录，默认使用 config.FIGURE_DIR
    """
    if save_dir is None:
        save_dir = config.FIGURE_DIR

    print('\n' + '=' * 60)
    print('数据集探索性分析 (EDA)')
    print('=' * 60)

    # 1. 类别分布
    print('\n[1/3] 绘制类别分布...')
    train_counts, test_counts = plot_class_distribution(
        save_path=os.path.join(save_dir, 'class_distribution.png')
    )

    # 2. 样本图像
    print('\n[2/3] 绘制样本图像...')
    plot_sample_images(
        samples_per_class=4,
        save_path=os.path.join(save_dir, 'sample_images.png')
    )

    # 3. 数据集摘要统计
    print('\n[3/3] 数据集摘要统计...')
    from torchvision import datasets as ds
    train_dataset = ds.ImageFolder(root=config.TRAIN_DIR)
    test_dataset = ds.ImageFolder(root=config.TEST_DIR)
    total = len(train_dataset) + len(test_dataset)

    print(f'\n  数据集总量: {total} 张')
    print(f'  训练集: {len(train_dataset)} 张 ({len(train_dataset)/total*100:.1f}%)')
    print(f'  测试集: {len(test_dataset)} 张 ({len(test_dataset)/total*100:.1f}%)')
    print(f'  类别数: {config.NUM_CLASSES}')
    print(f'  类别分布:')
    for i, name in enumerate(config.CLASS_DISPLAY_CN):
        t_count = train_counts[i]
        te_count = test_counts[i]
        print(f'    {name}: 训练 {t_count} + 测试 {te_count} = {t_count + te_count} '
              f'({(t_count + te_count)/total*100:.1f}%)')
    print(f'  图像尺寸: {config.IMG_SIZE}×{config.IMG_SIZE} (统一缩放后)')
    print('=' * 60)


if __name__ == '__main__':
    # 测试数据加载
    train_loader, val_loader, test_loader = create_dataloaders()
    # 获取一个批次查看形状
    images, labels = next(iter(train_loader))
    print(f'批次图像形状: {images.shape}, 标签形状: {labels.shape}')
