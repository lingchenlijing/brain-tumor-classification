# -*- coding: utf-8 -*-
"""
训练流程：包含完整的训练循环、混合精度训练、学习率调度、早停和检查点保存
"""
import os
import time
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import config
from utils import AverageMeter, EarlyStopping, save_checkpoint, load_checkpoint


def train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch):
    """
    训练一个 epoch

    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scaler: 混合精度 GradScaler
        device: 计算设备
        epoch: 当前 epoch 编号

    Returns:
        平均训练损失和训练准确率
    """
    model.train()
    loss_meter = AverageMeter('TrainLoss')
    acc_meter = AverageMeter('TrainAcc')
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch + 1} [训练]')
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type='cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)

        # 混合精度反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # 统计指标
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += batch_size
        acc = 100.0 * correct / total
        acc_meter.update(acc, batch_size)

        pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc:.2f}%'})

    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, criterion, device, epoch):
    """
    在验证集上评估模型

    Args:
        model: 模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        epoch: 当前 epoch 编号

    Returns:
        平均验证损失和验证准确率
    """
    model.eval()
    loss_meter = AverageMeter('ValLoss')
    correct = 0
    total = 0

    pbar = tqdm(val_loader, desc=f'Epoch {epoch + 1} [验证]')
    with torch.no_grad():
        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)

            batch_size = images.size(0)
            loss_meter.update(loss.item(), batch_size)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += batch_size

            acc = 100.0 * correct / total
            pbar.set_postfix({'loss': f'{loss_meter.avg:.4f}', 'acc': f'{acc:.2f}%'})

    val_acc = 100.0 * correct / total
    return loss_meter.avg, val_acc


def train(model, train_loader, val_loader, config_obj=None, resume_path=None):
    """
    完整训练流程

    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config_obj: 配置对象（默认使用 config 模块）
        resume_path: 检查点路径，用于恢复训练

    Returns:
        训练历史字典 {train_loss, val_loss, train_acc, val_acc}
    """
    if config_obj is None:
        config_obj = config

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # 损失函数（带标签平滑）
    criterion = nn.CrossEntropyLoss(label_smoothing=config_obj.LABEL_SMOOTHING)

    # 优化器（AdamW）
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config_obj.LR,
        weight_decay=config_obj.WEIGHT_DECAY
    )

    # 学习率调度器（余弦退火）
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config_obj.NUM_EPOCHS,
        eta_min=1e-6
    )

    # 混合精度训练
    scaler = GradScaler()

    # 早停
    early_stopping = EarlyStopping(patience=config_obj.EARLY_STOP_PATIENCE, mode='max')

    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
    }

    # 最佳模型追踪
    best_val_acc = 0.0
    start_epoch = 0
    model_name = model.__class__.__name__

    # 恢复训练
    if resume_path is not None and os.path.exists(resume_path):
        checkpoint = load_checkpoint(resume_path, model, optimizer)
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        history = checkpoint.get('history', history)
        print(f'从 epoch {start_epoch} 恢复训练，最佳验证准确率: {best_val_acc:.2f}%')

    # 检查点保存路径
    best_model_path = os.path.join(config_obj.MODEL_DIR, f'{model_name}_best.pth')
    last_model_path = os.path.join(config_obj.MODEL_DIR, f'{model_name}_last.pth')

    print(f'\n开始训练: {model_name}')
    print(f'设备: {device}')
    print(f'训练轮数: {config_obj.NUM_EPOCHS}')
    print(f'初始学习率: {config_obj.LR}')
    print('-' * 60)

    total_start = time.time()

    for epoch in range(start_epoch, config_obj.NUM_EPOCHS):
        epoch_start = time.time()

        # 训练
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device, epoch)

        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        # 记录历史
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start

        # 打印 epoch 总结
        print(f'\nEpoch {epoch + 1}/{config_obj.NUM_EPOCHS} | '
              f'训练损失: {train_loss:.4f} | 训练准确率: {train_acc:.2f}% | '
              f'验证损失: {val_loss:.4f} | 验证准确率: {val_acc:.2f}% | '
              f'学习率: {current_lr:.6f} | 耗时: {epoch_time:.1f}s')

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'history': history,
                },
                best_model_path
            )
            print(f'  ★ 最佳模型已保存 (验证准确率: {best_val_acc:.2f}%)')

        # 保存最新检查点
        save_checkpoint(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_acc': best_val_acc,
                'history': history,
            },
            last_model_path
        )

        # 早停检查
        if early_stopping(val_acc):
            print(f'\n训练提前停止于 epoch {epoch + 1}')
            break

    total_time = time.time() - total_start
    print(f'\n训练完成！总耗时: {total_time / 60:.1f} 分钟')
    print(f'最佳验证准确率: {best_val_acc:.2f}%')
    print(f'最佳模型保存于: {best_model_path}')

    return history


if __name__ == '__main__':
    # 快速测试训练流程
    from dataset import create_dataloaders
    from models import get_model
    from utils import set_seed, count_parameters

    set_seed(config.SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 使用小模型快速测试
    model = get_model('baseline')
    print(f'模型参数量: {count_parameters(model):,}')
    train_loader, val_loader, test_loader = create_dataloaders()
    history = train(model, train_loader, val_loader)
