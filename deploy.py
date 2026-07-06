# -*- coding: utf-8 -*-
"""
部署可行性分析：ONNX/TorchScript 导出、FLOPs 计算、推理性能基准、部署方案评估

对应任务书 CO3/CO4 要求：
  - 理解计算效率与资源限制的关系
  - 模型轻量化或部署的可行性方案
"""
import os
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils import count_parameters

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===========================================================================
#                           FLOPs 手动计算
# ===========================================================================

def compute_conv2d_flops(module, input_shape, output_shape):
    """计算 Conv2d 层的 FLOPs"""
    if module.groups == module.in_channels and module.in_channels == module.out_channels:
        # 深度可分离卷积
        flops_per_position = module.kernel_size[0] * module.kernel_size[1]
        flops = flops_per_position * output_shape[0] * output_shape[2] * output_shape[3]
    else:
        flops_per_position = (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
        flops = flops_per_position * output_shape[0] * output_shape[2] * output_shape[3] * module.out_channels

    # Bias 加法
    if module.bias is not None:
        flops += output_shape[0] * output_shape[2] * output_shape[3] * module.out_channels

    return flops


def compute_linear_flops(module):
    """计算 Linear 层的 FLOPs (乘加各算一次 = 2*in*out)"""
    return 2 * module.in_features * module.out_features + (module.out_features if module.bias is not None else 0)


def calculate_model_flops(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    手动逐层计算模型的 FLOPs（支持 ResNet 系列和 BaselineCNN）

    Returns:
        total_flops: 总 FLOPs
        layer_flops: 各层 FLOPs 列表
    """
    model = model.to(device)
    model.eval()

    total_flops = 0
    layer_flops = []
    hooks = []

    def _hook_conv(module, input, output):
        nonlocal total_flops
        flops = compute_conv2d_flops(module, input[0].shape, output.shape)
        total_flops += flops
        layer_flops.append({'name': module.__class__.__name__, 'flops': flops})

    def _hook_linear(module, input, output):
        nonlocal total_flops
        flops = compute_linear_flops(module)
        total_flops += flops
        layer_flops.append({'name': module.__class__.__name__, 'flops': flops})

    def _hook_bn(module, input, output):
        nonlocal total_flops
        # BatchNorm2d: 每个元素进行 (x-mean)/std * gamma + beta = 2次乘加
        flops = 2 * input[0].numel()
        total_flops += flops
        layer_flops.append({'name': 'BatchNorm2d', 'flops': flops})

    def _hook_relu(module, input, output):
        nonlocal total_flops
        # ReLU: 每个元素一次比较和可能的赋值
        flops = input[0].numel()
        total_flops += flops
        layer_flops.append({'name': 'ReLU', 'flops': flops})

    def _hook_avgpool(module, input, output):
        nonlocal total_flops
        # AdaptiveAvgPool: 每个输出元素需要 kernel_size 次加法 + 1次除法
        if hasattr(module, 'output_size'):
            out_size = module.output_size
            if isinstance(out_size, int):
                out_size = (out_size, out_size)
            spatial_out = out_size[0] * out_size[1]
            flops = input[0].shape[1] * spatial_out * 2
            total_flops += flops
            layer_flops.append({'name': 'AdaptiveAvgPool2d', 'flops': flops})

    def _hook_maxpool(module, input, output):
        nonlocal total_flops
        # MaxPool: 每个池化窗口一次比较
        flops = output.numel() * (module.kernel_size ** 2 if hasattr(module, 'kernel_size') else 4)
        total_flops += flops
        layer_flops.append({'name': 'MaxPool2d', 'flops': flops})

    # 注册钩子
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(_hook_conv))
        elif isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(_hook_linear))
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            hooks.append(module.register_forward_hook(_hook_bn))
        elif isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(_hook_relu))
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            hooks.append(module.register_forward_hook(_hook_avgpool))
        elif isinstance(module, nn.MaxPool2d):
            hooks.append(module.register_forward_hook(_hook_maxpool))

    # 前向传播触发钩子
    dummy_input = torch.randn(*input_size, device=device)
    with torch.no_grad():
        _ = model(dummy_input)

    # 移除钩子
    for h in hooks:
        h.remove()

    # 统计各类型层的 FLOPs
    flops_by_type = {}
    for entry in layer_flops:
        t = entry['name']
        flops_by_type[t] = flops_by_type.get(t, 0) + entry['flops']

    return total_flops, flops_by_type


def format_flops(flops):
    """格式化 FLOPs 为可读字符串"""
    if flops >= 1e9:
        return f'{flops / 1e9:.2f} GFLOPs'
    elif flops >= 1e6:
        return f'{flops / 1e6:.2f} MFLOPs'
    elif flops >= 1e3:
        return f'{flops / 1e3:.2f} KFLOPs'
    else:
        return f'{flops:.0f} FLOPs'


# ===========================================================================
#                           ONNX 导出
# ===========================================================================

def export_to_onnx(model, save_path, input_size=(1, 3, 224, 224), device='cuda',
                   dynamic_batch=True):
    """
    导出模型为 ONNX 格式

    Args:
        model: PyTorch 模型
        save_path: 保存路径
        input_size: 输入形状
        device: 计算设备
        dynamic_batch: 是否支持动态 batch size

    Returns:
        onnx_size_mb: ONNX 文件大小 (MB)
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)

    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input':  {0: 'batch_size'},
            'output': {0: 'batch_size'},
        }

    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes=dynamic_axes,
    )

    onnx_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'ONNX 模型已导出: {save_path} ({onnx_size_mb:.2f} MB)')

    return onnx_size_mb


# ===========================================================================
#                           TorchScript 导出
# ===========================================================================

def export_to_torchscript(model, save_path, input_size=(1, 3, 224, 224), device='cuda'):
    """
    导出模型为 TorchScript 格式（用于 C++ 部署或移动端）

    Args:
        model: PyTorch 模型
        save_path: 保存路径
        input_size: 输入形状
        device: 计算设备

    Returns:
        ts_size_mb: TorchScript 文件大小 (MB)
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)

    # 使用 trace 方式导出
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(save_path)

    ts_size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'TorchScript 模型已导出: {save_path} ({ts_size_mb:.2f} MB)')

    return ts_size_mb


# ===========================================================================
#                           GPU 显存占用估算
# ===========================================================================

def estimate_gpu_memory(model, input_size=(1, 3, 224, 224), device='cuda'):
    """
    估算模型在 GPU 上的显存占用（参数 + 激活值）
    """
    if device != 'cuda' or not torch.cuda.is_available():
        return {'param_memory_mb': 0, 'total_estimate_mb': 0, 'note': '无 GPU，无法估算'}

    model = model.to(device)
    model.train()

    # 参数显存（float32 = 4 bytes per param）
    param_bytes = sum(p.numel() * 4 for p in model.parameters())
    # 梯度显存（训练时需要，与参数量相同）
    grad_bytes = sum(p.numel() * 4 for p in model.parameters() if p.requires_grad)
    # 优化器状态 (Adam/AdamW 需要存储第一矩和第二矩，2x 参数量)
    optimizer_bytes = sum(p.numel() * 4 * 2 for p in model.parameters() if p.requires_grad)

    # 尝试实际测量激活值显存
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        dummy_input = torch.randn(*input_size, device=device)
        _ = model(dummy_input)

        activation_bytes = torch.cuda.max_memory_allocated(device)
        torch.cuda.empty_cache()
    except Exception:
        activation_bytes = 0

    param_mb = param_bytes / (1024 ** 2)
    grad_mb = grad_bytes / (1024 ** 2)
    optimizer_mb = optimizer_bytes / (1024 ** 2)
    activation_mb = activation_bytes / (1024 ** 2)

    total_train_mb = param_mb + grad_mb + optimizer_mb + activation_mb
    total_infer_mb = param_mb + activation_mb

    return {
        'param_memory_mb':     param_mb,
        'gradient_memory_mb':  grad_mb,
        'optimizer_memory_mb': optimizer_mb,
        'activation_memory_mb': activation_mb,
        'total_train_mb':      total_train_mb,
        'total_infer_mb':      total_infer_mb,
    }


# ===========================================================================
#                          综合部署可行性报告
# ===========================================================================

def run_deployment_analysis(model, model_name='model', device='cuda'):
    """
    运行完整的部署可行性分析，生成综合报告

    包括：FLOPs、参数量、模型大小、推理速度、ONNX/TorchScript 导出、GPU 显存估算
    """
    print('\n' + '#' * 70)
    print('#  部署可行性综合分析')
    print('#' * 70)

    model.eval()
    model = model.to(device)
    input_size = (1, 3, config.IMG_SIZE, config.IMG_SIZE)

    report = {}

    # ---- 1. 基础信息 ----
    print('\n[1/6] 模型基础信息...')
    num_params = count_parameters(model)
    report['params'] = num_params
    print(f'  可训练参数量: {num_params:,}')
    print(f'  模型类名: {model.__class__.__name__}')

    # ---- 2. FLOPs ----
    print('\n[2/6] 计算 FLOPs...')
    try:
        total_flops, flops_by_type = calculate_model_flops(model, input_size, device)
        report['flops'] = total_flops
        report['flops_by_type'] = flops_by_type
        print(f'  总 FLOPs: {format_flops(total_flops)}')
        print('  各类型层分布:')
        for layer_type, flops in sorted(flops_by_type.items(), key=lambda x: x[1], reverse=True):
            pct = flops / total_flops * 100
            print(f'    {layer_type:<20}: {format_flops(flops):>12} ({pct:.1f}%)')
    except Exception as e:
        print(f'  FLOPs 计算失败: {e}')
        report['flops'] = 0
        report['flops_by_type'] = {}

    # ---- 3. 推理速度基准 ----
    print('\n[3/6] 推理速度基准测试...')
    from optimize import benchmark_inference_speed

    bench_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        speed = benchmark_inference_speed(model, input_size=input_size, device=bench_device)
        report['speed'] = speed
        print(f'  设备: {speed["device"]}')
        print(f'  平均延迟: {speed["mean_latency_ms"]:.2f} ms ± {speed["std_latency_ms"]:.2f}')
        print(f'  吞吐量: {speed["throughput_fps"]:.1f} fps')
    except Exception as e:
        print(f'  基准测试失败: {e}')
        report['speed'] = None

    # ---- 4. 模型大小 ----
    print('\n[4/6] 模型存储大小...')
    tmp_path = os.path.join(config.MODEL_DIR, f'_tmp_{model_name}.pth')
    pth_size = 0
    try:
        torch.save(model.state_dict(), tmp_path)
        pth_size = os.path.getsize(tmp_path) / (1024 ** 2)
        os.remove(tmp_path)
    except Exception:
        pass
    report['pth_size_mb'] = pth_size
    print(f'  PyTorch 权重文件大小: {pth_size:.2f} MB')

    # ---- 5. ONNX / TorchScript 导出 ----
    print('\n[5/6] 模型格式导出...')

    onnx_path = os.path.join(config.MODEL_DIR, f'{model_name}.onnx')
    try:
        onnx_size = export_to_onnx(model, onnx_path, input_size, device)
        report['onnx_size_mb'] = onnx_size
        report['onnx_path'] = onnx_path
    except Exception as e:
        print(f'  ONNX 导出失败: {e}')
        report['onnx_size_mb'] = 0
        report['onnx_path'] = None

    ts_path = os.path.join(config.MODEL_DIR, f'{model_name}.pt')
    try:
        ts_size = export_to_torchscript(model, ts_path, input_size, device)
        report['ts_size_mb'] = ts_size
        report['ts_path'] = ts_path
    except Exception as e:
        print(f'  TorchScript 导出失败: {e}')
        report['ts_size_mb'] = 0
        report['ts_path'] = None

    # ---- 6. GPU 显存估算 ----
    print('\n[6/6] GPU 显存占用估算...')
    memory_info = estimate_gpu_memory(model, input_size, device)
    report['memory'] = memory_info
    if memory_info.get('note'):
        print(f'  {memory_info["note"]}')
    else:
        print(f'  参数显存: {memory_info["param_memory_mb"]:.1f} MB')
        print(f'  梯度显存: {memory_info["gradient_memory_mb"]:.1f} MB')
        print(f'  优化器状态: {memory_info["optimizer_memory_mb"]:.1f} MB')
        print(f'  激活值: {memory_info["activation_memory_mb"]:.1f} MB')
        print(f'  训练总占用 (估算): {memory_info["total_train_mb"]:.1f} MB')
        print(f'  推理总占用 (估算): {memory_info["total_infer_mb"]:.1f} MB')

    # ---- 综合总结 ----
    print_deployment_summary(report, model_name)

    # ---- 可视化 ----
    plot_deployment_report(report, model_name)

    print('#' * 70)
    return report


def print_deployment_summary(report, model_name):
    """打印部署可行性综合总结"""
    print('\n' + '=' * 70)
    print('  部署可行性综合评估')
    print('=' * 70)

    params = report['params']
    flops = report['flops']
    speed = report['speed']
    mem = report['memory']

    print(f'\n  模型: {model_name}')
    print(f'  参数量: {params:,}')
    print(f'  计算量: {format_flops(flops)}')
    if report.get('pth_size_mb', 0) > 0:
        print(f'  权重文件: {report["pth_size_mb"]:.1f} MB')
    if report.get('onnx_size_mb', 0) > 0:
        print(f'  ONNX 文件: {report["onnx_size_mb"]:.1f} MB')
    if report.get('ts_size_mb', 0) > 0:
        print(f'  TorchScript: {report["ts_size_mb"]:.1f} MB')

    if speed:
        print(f'  推理延迟: {speed["mean_latency_ms"]:.2f} ms')
        print(f'  吞吐量: {speed["throughput_fps"]:.1f} fps')

    # 部署场景评估
    print('\n  --- 部署场景评估 ---')

    # 移动端/边缘设备
    if params < 5_000_000 and flops < 500_000_000:
        print('  ✅ 移动端/边缘设备: 模型轻量，可部署到移动设备或嵌入式平台')
    elif params < 25_000_000 and flops < 5_000_000_000:
        print('  ⚠️  移动端/边缘设备: 需经剪枝量化后方可部署')
    else:
        print('  ❌ 移动端/边缘设备: 模型过大，建议使用轻量级替代架构')

    # 云端服务器
    if params < 100_000_000:
        print('  ✅ 云端服务器: 可轻松部署，单 GPU 即可承载')
    else:
        print('  ⚠️  云端服务器: 需要高性能 GPU 进行推理')

    # 实时推理（>30fps）
    if speed and speed['throughput_fps'] > 30:
        print('  ✅ 实时推理 (>30fps): 满足实时应用需求')
    elif speed and speed['throughput_fps'] > 10:
        print('  ⚠️  实时推理: 基本满足准实时需求，可进一步优化')
    else:
        print('  ❌ 实时推理: 当前延迟无法满足实时需求')

    # ONNX 兼容性
    if report.get('onnx_size_mb', 0) > 0:
        print(f'  ✅ ONNX 格式已导出: {report["onnx_path"]}')
        print('     -> 可部署到 ONNX Runtime / TensorRT / OpenVINO 等推理后端')
    if report.get('ts_size_mb', 0) > 0:
        print(f'  ✅ TorchScript 已导出: {report["ts_path"]}')
        print('     -> 可用于 C++ LibTorch 部署或移动端 PyTorch Mobile')

    print('=' * 70)


def plot_deployment_report(report, model_name, save_path=None):
    """绘制部署报告可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ---- 图1: 模型大小对比（不同格式） ----
    formats = ['PyTorch\n(.pth)', 'ONNX\n(.onnx)', 'TorchScript\n(.pt)']
    sizes = [
        report.get('pth_size_mb', 0),
        report.get('onnx_size_mb', 0),
        report.get('ts_size_mb', 0),
    ]
    colors_fmt = ['#2196F3', '#FF9800', '#4CAF50']
    bars = axes[0].bar(formats, sizes, color=colors_fmt, alpha=0.85, edgecolor='white', width=0.5)
    for bar, s in zip(bars, sizes):
        if s > 0:
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{s:.1f} MB', ha='center', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('文件大小 (MB)')
    axes[0].set_title('不同格式模型大小', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')

    # ---- 图2: FLOPs 按层类型分布 ----
    if report.get('flops_by_type'):
        layer_types = list(report['flops_by_type'].keys())
        flops_vals = list(report['flops_by_type'].values())
        colors_pie = plt.cm.Set3(np.linspace(0, 1, len(layer_types)))
        wedges, texts, autotexts = axes[1].pie(
            flops_vals, labels=layer_types, autopct='%1.1f%%',
            colors=colors_pie, textprops={'fontsize': 8}
        )
        axes[1].set_title(f'FLOPs 按层类型分布\n总计: {format_flops(report["flops"])}', fontsize=12)

    # ---- 图3: GPU 显存占用分解 ----
    if report.get('memory') and report['memory'].get('total_train_mb', 0) > 0:
        mem = report['memory']
        mem_items = ['参数', '梯度', '优化器', '激活值']
        mem_vals = [
            mem.get('param_memory_mb', 0),
            mem.get('gradient_memory_mb', 0),
            mem.get('optimizer_memory_mb', 0),
            mem.get('activation_memory_mb', 0),
        ]
        mem_colors = ['#4CAF50', '#FF9800', '#E53935', '#2196F3']
        bars2 = axes[2].bar(mem_items, mem_vals, color=mem_colors, alpha=0.85, edgecolor='white', width=0.5)
        for bar, v in zip(bars2, mem_vals):
            if v > 0:
                axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                             f'{v:.0f} MB', ha='center', fontsize=9, fontweight='bold')
        axes[2].set_ylabel('显存 (MB)')
        axes[2].set_title(f'训练显存占用分解\n总训练占用 ≈ {mem["total_train_mb"]:.0f} MB', fontsize=12)
        axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, f'{model_name}_deployment_report.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'部署分析图已保存至: {save_path}')


# ===========================================================================
#                          独立脚本入口
# ===========================================================================

if __name__ == '__main__':
    from dataset import create_dataloaders
    from models import get_model
    from utils import set_seed, get_device

    set_seed(config.SEED)
    device = get_device()

    model = get_model('resnet50_cbam')

    # 尝试加载最佳模型
    best_path = os.path.join(config.MODEL_DIR, 'ResNet_best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print('已加载最佳模型')

    report = run_deployment_analysis(model, model_name='resnet50_cbam', device=str(device))
