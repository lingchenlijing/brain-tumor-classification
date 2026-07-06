# -*- coding: utf-8 -*-
"""
模型优化与对比：结构化剪枝、动态量化（INT8）、优化前后性能对比

对应任务书 CO3/CO4 要求：
  - 针对模型瓶颈提出至少一种优化方案，展示优化前后对比结果
  - 模型轻量化或部署的可行性方案
"""
import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torch.quantization import quantize_dynamic
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import config
from utils import count_parameters

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ===========================================================================
#                        结构化剪枝（按通道 L1 范数）
# ===========================================================================

def apply_structured_pruning(model, pruning_ratio=0.2, target_layer_types=(nn.Conv2d,)):
    """
    对模型中指定类型的层应用结构化剪枝（按通道L1范数），移除不重要的通道。

    剪枝策略：对每个 Conv2d 层，移除 L1 范数最小的 (pruning_ratio * out_channels) 个滤波器。

    Args:
        model: PyTorch 模型
        pruning_ratio: 剪枝比例 (0.0 ~ 1.0)
        target_layer_types: 需要剪枝的层类型

    Returns:
        pruned_model: 剪枝后的模型（深拷贝）
        prune_info: 剪枝详情字典
    """
    pruned_model = copy.deepcopy(model)
    prune_info = {'total_channels': 0, 'pruned_channels': 0, 'layer_details': []}

    for name, module in pruned_model.named_modules():
        if isinstance(module, target_layer_types):
            weight = module.weight.data
            out_channels = weight.size(0)
            num_prune = int(out_channels * pruning_ratio)

            if num_prune == 0:
                continue

            # 计算每个输出通道的 L1 范数
            l1_norms = weight.abs().sum(dim=(1, 2, 3))
            _, prune_indices = torch.topk(l1_norms, num_prune, largest=False)

            # 创建剪枝掩码
            mask = torch.ones(out_channels, device=weight.device)
            mask[prune_indices] = 0

            # 应用掩码（输出通道）
            module.weight.data = weight * mask[:, None, None, None]
            if module.bias is not None:
                module.bias.data = module.bias.data * mask

            prune_info['total_channels'] += out_channels
            prune_info['pruned_channels'] += num_prune
            prune_info['layer_details'].append({
                'name': name,
                'out_channels': out_channels,
                'pruned': num_prune,
                'ratio': pruning_ratio,
            })

    actual_ratio = prune_info['pruned_channels'] / max(prune_info['total_channels'], 1) * 100
    print(f'结构化剪枝完成: 移除 {prune_info["pruned_channels"]}/{prune_info["total_channels"]} '
          f'个卷积通道 ({actual_ratio:.1f}%)')

    return pruned_model, prune_info


# ===========================================================================
#                           动态量化（INT8）
# ===========================================================================

def apply_dynamic_quantization(model, dtype=torch.qint8):
    """
    对模型应用动态量化（INT8），主要量化 nn.Linear 层。

    PyTorch 动态量化在推理时将权重转换为 INT8，激活保持浮点，
    在每次前向传播时动态计算量化参数。对 CPU 推理加速效果显著。

    Args:
        model: PyTorch 模型（CPU）
        dtype: 量化数据类型 (默认 torch.qint8)

    Returns:
        quantized_model: 量化后的模型
    """
    model_cpu = copy.deepcopy(model).cpu()

    # 对 Linear 层应用动态量化
    quantized_model = quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Conv2d},
        dtype=dtype
    )

    print('动态量化 (INT8) 完成')

    return quantized_model


# ===========================================================================
#                          推理速度基准测试
# ===========================================================================

def benchmark_inference_speed(model, input_size=(1, 3, 224, 224), device='cuda',
                              num_warmup=10, num_runs=100):
    """
    测量模型的推理延迟（吞吐量）

    Args:
        model: PyTorch 模型
        input_size: 输入张量形状 (B, C, H, W)
        device: 计算设备
        num_warmup: 预热运行次数
        num_runs: 正式测量运行次数

    Returns:
        benchmark: 包含延迟和吞吐量指标的字典
    """
    model = model.to(device)
    model.eval()

    dummy_input = torch.randn(*input_size, device=device)

    # 预热 GPU
    if device == 'cuda':
        torch.cuda.synchronize()

    for _ in range(num_warmup):
        with torch.no_grad():
            _ = model(dummy_input)

    if device == 'cuda':
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 正式测量
    latencies = []
    if device == 'cuda':
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        for _ in range(num_runs):
            starter.record()
            with torch.no_grad():
                _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            latencies.append(starter.elapsed_time(ender))  # ms
    else:
        for _ in range(num_runs):
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(dummy_input)
            latencies.append((time.perf_counter() - start) * 1000)  # ms

    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)
    throughput = 1000.0 / mean_latency  # samples/second

    return {
        'mean_latency_ms': mean_latency,
        'std_latency_ms':  std_latency,
        'throughput_fps':  throughput,
        'num_runs':        num_runs,
        'device':          device,
    }


# ===========================================================================
#                          模型保存与大小测量
# ===========================================================================

def get_model_size_mb(model, save_path=None):
    """
    获取模型参数量对应的文件大小（MB）

    Args:
        model: PyTorch 模型
        save_path: 如果不为 None，保存模型到此路径并返回实际文件大小

    Returns:
        size_mb: 模型大小（MB）
    """
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        size_bytes = os.path.getsize(save_path)
        os.remove(save_path)  # 清理临时文件
    else:
        # 估算：每个 float32 参数占 4 字节
        num_params = count_parameters(model)
        size_bytes = num_params * 4

    return size_bytes / (1024 * 1024)


def export_and_measure_onnx(model, dummy_input, save_path):
    """
    导出 ONNX 并返回文件大小
    """
    model.eval()
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=14,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    print(f'ONNX 模型已导出至: {save_path} ({size_mb:.2f} MB)')
    return size_mb


# ===========================================================================
#                          完整优化流水线
# ===========================================================================

def run_optimization_pipeline(model, test_loader, device, prune_ratios=(0.1, 0.2, 0.3),
                              model_name='model'):
    """
    运行完整优化流水线：
      1. 原始模型基准测试
      2. 结构化剪枝 (3个比例) + 评估
      3. 动态量化 + 评估
      4. 生成优化前后对比结果

    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
        prune_ratios: 需要尝试的剪枝比例
        model_name: 模型名称

    Returns:
        comparison_table: 优化前后对比表
    """
    from evaluate import get_predictions, compute_metrics

    print('\n' + '#' * 70)
    print('#  模型优化流水线 — 剪枝 + 量化 + 性能对比')
    print('#' * 70)

    results = []

    # ---- Step 1: 原始模型基准 ----
    print('\n[1/5] 原始模型基准测试...')
    model.eval()
    original_params = count_parameters(model)
    original_size_mb = get_model_size_mb(model)

    labels, preds, probs = get_predictions(model, test_loader, device)
    original_metrics = compute_metrics(labels, preds, probs)

    bench_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        original_speed = benchmark_inference_speed(model, device=bench_device)
    except Exception as e:
        print(f'  GPU 基准测试失败（{e}），使用 CPU')
        original_speed = benchmark_inference_speed(model, device='cpu')

    results.append({
        'name':           '原始模型 (FP32)',
        'params':         original_params,
        'size_mb':        original_size_mb,
        'accuracy':       original_metrics['accuracy'],
        'f1_macro':       original_metrics['f1_macro'],
        'f1_weighted':    original_metrics['f1_weighted'],
        'latency_ms':     original_speed['mean_latency_ms'],
        'throughput_fps': original_speed['throughput_fps'],
    })

    print(f'  原始模型: 参数量={original_params:,}, 大小≈{original_size_mb:.2f} MB, '
          f'准确率={original_metrics["accuracy"]:.4f}, '
          f'延迟={original_speed["mean_latency_ms"]:.2f} ms')

    # ---- Step 2: 结构化剪枝实验 ----
    for ratio in prune_ratios:
        print(f'\n[2/5] 结构化剪枝 (ratio={ratio:.0%})...')

        pruned_model, prune_info = apply_structured_pruning(
            model, pruning_ratio=ratio
        )
        pruned_model = pruned_model.to(device)

        # 评估精度
        labels_p, preds_p, probs_p = get_predictions(pruned_model, test_loader, device)
        metrics_p = compute_metrics(labels_p, preds_p, probs_p)
        speed_p = benchmark_inference_speed(pruned_model, device=bench_device)

        results.append({
            'name':           f'结构化剪枝 ({ratio:.0%})',
            'params':         count_parameters(pruned_model),
            'size_mb':        get_model_size_mb(pruned_model),
            'accuracy':       metrics_p['accuracy'],
            'f1_macro':       metrics_p['f1_macro'],
            'f1_weighted':    metrics_p['f1_weighted'],
            'latency_ms':     speed_p['mean_latency_ms'],
            'throughput_fps': speed_p['throughput_fps'],
        })

        print(f'  剪枝 {ratio:.0%}: 准确率={metrics_p["accuracy"]:.4f}, '
              f'延迟={speed_p["mean_latency_ms"]:.2f} ms')

        del pruned_model
        if device == 'cuda':
            torch.cuda.empty_cache()

    # ---- Step 3: 模型量化（INT8） ----
    print(f'\n[3/5] 动态量化 (INT8)...')

    try:
        quantized_model = apply_dynamic_quantization(model)
        quantized_model = quantized_model.to('cpu')

        # 量化模型在 CPU 上评估
        labels_q, preds_q, probs_q = get_predictions(quantized_model, test_loader, 'cpu')
        metrics_q = compute_metrics(labels_q, preds_q, probs_q)
        speed_q = benchmark_inference_speed(quantized_model, device='cpu')

        # 量化模型大小（实际文件大小）
        tmp_path = os.path.join(config.MODEL_DIR, '_tmp_quantized.pth')
        size_q = get_model_size_mb(quantized_model, save_path=tmp_path)

        results.append({
            'name':           '动态量化 (INT8)',
            'params':         count_parameters(quantized_model),
            'size_mb':        size_q,
            'accuracy':       metrics_q['accuracy'],
            'f1_macro':       metrics_q['f1_macro'],
            'f1_weighted':    metrics_q['f1_weighted'],
            'latency_ms':     speed_q['mean_latency_ms'],
            'throughput_fps': speed_q['throughput_fps'],
        })

        print(f'  量化 INT8: 模型大小={size_q:.2f} MB, 准确率={metrics_q["accuracy"]:.4f}, '
              f'延迟={speed_q["mean_latency_ms"]:.2f} ms')

        del quantized_model
    except Exception as e:
        print(f'  量化评估失败: {e}，跳过量化实验')
        # 添加占位项
        results.append({
            'name':           '动态量化 (INT8)',
            'params':         0,
            'size_mb':        0,
            'accuracy':       0,
            'f1_macro':       0,
            'f1_weighted':    0,
            'latency_ms':     0,
            'throughput_fps': 0,
        })

    # ---- Step 4: 打印汇总表 ----
    print_optimization_summary(results)

    # ---- Step 5: 绘制对比图 ----
    plot_optimization_comparison(
        results,
        save_path=os.path.join(config.FIGURE_DIR, f'{model_name}_optimization_comparison.png')
    )

    print('#' * 70)
    return results


def print_optimization_summary(results):
    """打印优化前后对比汇总表"""
    print(f'\n{"=" * 100}')
    print('  模型优化前后对比汇总')
    print(f'{"=" * 100}')
    header = (f'{"方案":<22} {"参数量":>10} {"大小MB":>8} {"准确率":>8} '
              f'{"F1(加权)":>10} {"延迟ms":>8} {"吞吐fps":>8} {"精度变化":>8}')
    print(header)
    print('-' * 100)

    baseline_acc = results[0]['accuracy']
    baseline_speed = results[0]['latency_ms']

    for r in results:
        acc_delta = r['accuracy'] - baseline_acc if r['accuracy'] > 0 else 0
        acc_str = f'{acc_delta:+.4f}' if acc_delta != 0 else '-'
        speed_delta = ((baseline_speed - r['latency_ms']) / baseline_speed * 100
                       if r['latency_ms'] > 0 and baseline_speed > 0 else 0)
        speed_str = f'{speed_delta:+.1f}%' if speed_delta != 0 else '-'

        print(f'{r["name"]:<22} {r["params"]:>10,} {r["size_mb"]:>7.2f} {r["accuracy"]:>8.4f} '
              f'{r["f1_weighted"]:>10.4f} {r["latency_ms"]:>7.2f} {r["throughput_fps"]:>7.1f} '
              f'{acc_str:>8}')

    print(f'{"=" * 100}')

    # 推荐最优方案
    valid_results = [r for r in results[1:] if r['accuracy'] > 0]
    if valid_results:
        # 找精度损失最小且速度提升最大的方案
        best = min(valid_results,
                   key=lambda r: (baseline_acc - r['accuracy']) * 100 + r['latency_ms'] * 0.01)
        print(f'\n推荐优化方案: {best["name"]}')
        if best['accuracy'] > 0:
            acc_loss = (baseline_acc - best['accuracy']) * 100
            print(f'  精度损失: {acc_loss:.2f}%')
            if best['latency_ms'] > 0 and baseline_speed > 0:
                speedup = baseline_speed / best['latency_ms']
                print(f'  推理加速: {speedup:.2f}x')


def plot_optimization_comparison(results, save_path=None):
    """
    绘制优化对比图：准确率 vs 推理速度的帕累托前沿
    """
    valid = [r for r in results if r['accuracy'] > 0]
    if len(valid) < 2:
        print('优化实验数据不足，跳过图表绘制')
        return

    names = [r['name'] for r in valid]
    accuracies = [r['accuracy'] * 100 for r in valid]
    latencies = [r['latency_ms'] for r in valid]
    sizes = [r['size_mb'] for r in valid]
    fps_vals = [r['throughput_fps'] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#2196F3', '#FF9800', '#FF9800', '#FF9800', '#4CAF50']

    # ---- 图1: 准确率 vs 延迟 (帕累托) ----
    for i in range(len(valid)):
        axes[0].scatter(latencies[i], accuracies[i], s=200, c=colors[i % len(colors)],
                        alpha=0.85, edgecolors='white', lw=1.5, zorder=5)
        axes[0].annotate(names[i].replace(' ', '\n'),
                         (latencies[i], accuracies[i]),
                         xytext=(8, 4), textcoords='offset points', fontsize=7)

    axes[0].set_xlabel('推理延迟 (ms)', fontsize=10)
    axes[0].set_ylabel('准确率 (%)', fontsize=10)
    axes[0].set_title('准确率 vs 推理延迟 (帕累托前沿)', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # ---- 图2: 模型大小对比 ----
    bar_colors = colors[:len(valid)]
    bars = axes[1].barh(range(len(valid)), sizes, color=bar_colors, alpha=0.85, edgecolor='white')
    for i, (bar, s) in enumerate(zip(bars, sizes)):
        axes[1].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{s:.1f} MB', va='center', fontsize=8)
    axes[1].set_yticks(range(len(valid)))
    axes[1].set_yticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    axes[1].set_xlabel('模型大小 (MB)', fontsize=10)
    axes[1].set_title('模型存储大小对比', fontsize=12)
    axes[1].invert_yaxis()

    # ---- 图3: 吞吐量对比 ----
    bars2 = axes[2].barh(range(len(valid)), fps_vals, color=bar_colors, alpha=0.85, edgecolor='white')
    for i, (bar, f) in enumerate(zip(bars2, fps_vals)):
        axes[2].text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                     f'{f:.1f} fps', va='center', fontsize=8)
    axes[2].set_yticks(range(len(valid)))
    axes[2].set_yticklabels([n.replace(' ', '\n') for n in names], fontsize=7)
    axes[2].set_xlabel('吞吐量 (fps)', fontsize=10)
    axes[2].set_title('推理吞吐量对比', fontsize=12)
    axes[2].invert_yaxis()

    plt.tight_layout()

    if save_path is None:
        save_path = os.path.join(config.FIGURE_DIR, 'optimization_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'优化对比图已保存至: {save_path}')


# ===========================================================================
#                          独立脚本入口
# ===========================================================================

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

    results = run_optimization_pipeline(model, test_loader, device, model_name='resnet50_cbam')
