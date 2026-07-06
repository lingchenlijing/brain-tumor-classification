# -*- coding: utf-8 -*-
"""
主入口：解析命令行参数，执行训练、消融实验、模型优化或部署分析
用法:
  python run.py --model resnet50_cbam              # 训练指定模型
  python run.py --model baseline --epochs 20       # 自定义训练轮数
  python run.py --ablation                         # 运行消融实验
  python run.py --optimize                         # 运行模型优化流水线（剪枝+量化）
  python run.py --deploy                           # 运行部署可行性分析
  python run.py --full                             # 全流程：训练+评估+偏差分析+优化+部署
"""
import argparse
import os
import torch
import config
from utils import set_seed, get_device, count_parameters
from dataset import create_dataloaders
from models import get_model
from train import train
from evaluate import evaluate_model
from ablation import run_ablation_study


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='脑肿瘤分类 - 深度学习项目')
    parser.add_argument('--model', type=str, default='resnet50_cbam',
                        choices=config.MODEL_NAMES,
                        help='模型名称 (default: resnet50_cbam)')
    parser.add_argument('--epochs', type=int, default=None,
                        help=f'训练轮数 (default: {config.NUM_EPOCHS})')
    parser.add_argument('--lr', type=float, default=None,
                        help=f'学习率 (default: {config.LR})')
    parser.add_argument('--batch_size', type=int, default=None,
                        help=f'批大小 (default: {config.BATCH_SIZE})')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验（忽略 --model 参数）')
    parser.add_argument('--no_aug', action='store_true',
                        help='不使用数据增强')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练的路径')
    parser.add_argument('--optimize', action='store_true',
                        help='运行模型优化流水线（结构化剪枝 + INT8 量化对比）')
    parser.add_argument('--deploy', action='store_true',
                        help='运行部署可行性分析（ONNX导出/FLOPs/推理基准/显存估算）')
    parser.add_argument('--full', action='store_true',
                        help='全流程：训练 + 评估 + 偏差分析 + 优化 + 部署分析')
    parser.add_argument('--skip_train', action='store_true',
                        help='跳过训练，直接加载已有模型进行评估/优化/部署')
    parser.add_argument('--eda', action='store_true',
                        help='运行数据集EDA可视化（类别分布图+样本图像）')
    return parser.parse_args()


def load_best_model(model, device, model_class_name=None):
    """加载最佳模型检查点"""
    if model_class_name is None:
        model_class_name = model.__class__.__name__
    best_model_path = os.path.join(config.MODEL_DIR, f'{model_class_name}_best.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f'已加载最佳模型: {best_model_path}')
        return True
    else:
        print(f'⚠ 未找到检查点: {best_model_path}，使用未训练的模型')
        return False


def main():
    """主函数"""
    args = parse_args()

    # 设置随机种子
    set_seed(config.SEED)

    # 获取设备
    device = get_device()

    # 覆盖配置参数
    if args.epochs is not None:
        config.NUM_EPOCHS = args.epochs
    if args.lr is not None:
        config.LR = args.lr
    if args.batch_size is not None:
        config.BATCH_SIZE = args.batch_size

    # ---- EDA 模式 ----
    if args.eda or args.full:
        from dataset import run_eda
        run_eda()

    # ---- 消融实验模式 ----
    if args.ablation:
        print('\n' + '=' * 60)
        print('运行消融实验')
        print('=' * 60)
        results = run_ablation_study()
        return

    # ---- 全流程模式 ----
    if args.full:
        print('\n' + '#' * 70)
        print('#  全流程运行：EDA → 训练 → 评估 → 偏差分析 → 优化 → 部署')
        print('#' * 70)
        args.skip_train = False
        args.optimize = True
        args.deploy = True

    # ---- 训练模式 ----
    if not args.skip_train and not args.optimize and not args.deploy:
        # 纯训练模式
        print('\n' + '=' * 60)
        print(f'脑肿瘤分类 - 模型: {args.model}')
        print('=' * 60)

    if not args.skip_train:
        # 创建数据加载器
        use_augmentation = not args.no_aug
        train_loader, val_loader, test_loader = create_dataloaders(
            use_augmentation=use_augmentation,
            batch_size=config.BATCH_SIZE
        )

        # 创建模型
        model = get_model(args.model)
        print(f'\n模型: {model.__class__.__name__}')
        print(f'可训练参数量: {count_parameters(model):,}')
        print(f'数据增强: {use_augmentation}')
        print(f'训练轮数: {config.NUM_EPOCHS}')
        print(f'学习率: {config.LR}')
        print(f'批大小: {config.BATCH_SIZE}')

        # 训练
        history = train(model, train_loader, val_loader, resume_path=args.resume)
        model = model.to(device)

        # 加载最佳模型
        model_class_name = model.__class__.__name__
        load_best_model(model, device, model_class_name)

        # 评估（含子群体偏差分析）
        metrics = evaluate_model(
            model, test_loader, device,
            history=history,
            model_name=args.model
        )

        # 打印最终摘要
        print('\n' + '=' * 60)
        print('训练与评估完成 - 结果摘要')
        print('=' * 60)
        print(f'模型: {args.model}')
        print(f'测试准确率: {metrics["accuracy"]:.4f}')
        print(f'测试 F1 (macro): {metrics["f1_macro"]:.4f}')
        print(f'测试 F1 (weighted): {metrics["f1_weighted"]:.4f}')
        if 'auc_macro' in metrics:
            print(f'测试 AUC (macro): {metrics["auc_macro"]:.4f}')
        print(f'模型保存于: {config.MODEL_DIR}')
        print(f'图表保存于: {config.FIGURE_DIR}')
        print('=' * 60)

    else:
        # 跳过训练：直接加载数据加载器和模型
        _, _, test_loader = create_dataloaders(
            use_augmentation=not args.no_aug,
            batch_size=config.BATCH_SIZE
        )
        model = get_model(args.model)
        model = model.to(device)
        load_best_model(model, device)

    # ---- 模型优化模式 ----
    if args.optimize:
        from optimize import run_optimization_pipeline
        print('\n')
        results = run_optimization_pipeline(model, test_loader, device, model_name=args.model)

    # ---- 部署分析模式 ----
    if args.deploy:
        from deploy import run_deployment_analysis
        print('\n')
        report = run_deployment_analysis(model, model_name=args.model, device=str(device))


if __name__ == '__main__':
    main()
