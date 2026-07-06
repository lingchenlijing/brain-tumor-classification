# -*- coding: utf-8 -*-
"""
模型定义：BaselineCNN、ResNet18/50 预训练、ResNet50_CBAM（含 CBAM 注意力机制）
CBAM = Convolutional Block Attention Module（通道注意力 + 空间注意力）
"""
import torch
import torch.nn as nn
import torchvision.models as models
import config


# ======================== 通道注意力模块 ========================
class ChannelAttention(nn.Module):
    """
    通道注意力模块：通过平均池化和最大池化提取通道信息，
    经共享 MLP 后用 sigmoid 生成通道权重
    """

    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        # 共享 MLP 网络
        mid_channels = max(in_channels // reduction_ratio, 1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, mid_channels, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, in_channels, bias=True)
        )

    def forward(self, x):
        # x 形状: (B, C, H, W)
        b, c, h, w = x.shape
        # 平均池化和最大池化（沿空间维度）
        avg_pool = nn.functional.adaptive_avg_pool2d(x, 1).view(b, c)  # (B, C)
        max_pool = nn.functional.adaptive_max_pool2d(x, 1).view(b, c)  # (B, C)
        # 共享 MLP
        avg_out = self.mlp(avg_pool)  # (B, C)
        max_out = self.mlp(max_pool)  # (B, C)
        # 相加后 sigmoid
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)  # (B, C, 1, 1)
        return x * attention


# ======================== 空间注意力模块 ========================
class SpatialAttention(nn.Module):
    """
    空间注意力模块：沿通道维度做平均池化和最大池化，
    拼接后经卷积和 sigmoid 生成空间权重
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(1)

    def forward(self, x):
        # x 形状: (B, C, H, W)
        # 沿通道维度做平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)   # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B, 1, H, W)
        # 拼接后卷积
        concat = torch.cat([avg_out, max_out], dim=1)  # (B, 2, H, W)
        attention = torch.sigmoid(self.bn(self.conv(concat)))  # (B, 1, H, W)
        return x * attention


# ======================== CBAM 模块 ========================
class CBAMBlock(nn.Module):
    """
    CBAM 模块 = 通道注意力 + 空间注意力
    依次施加通道注意力和空间注意力
    """

    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMBlock, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# ======================== 基线 CNN ========================
class BaselineCNN(nn.Module):
    """
    基线 CNN：4 层 Conv-BN-ReLU-Pool + 全连接层
    约 2M 可训练参数
    """

    def __init__(self, num_classes=4):
        super(BaselineCNN, self).__init__()
        self.features = nn.Sequential(
            # 第1层
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 224 -> 112

            # 第2层
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 112 -> 56

            # 第3层
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 56 -> 28

            # 第4层
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),       # 28 -> 14
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 全局平均池化 -> (512, 1, 1)
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ======================== ResNet18 预训练 ========================
def resnet18_pretrained(num_classes=4):
    """加载 ImageNet 预训练的 ResNet18，修改最后的全连接层"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ======================== ResNet50 预训练 ========================
def resnet50_pretrained(num_classes=4):
    """加载 ImageNet 预训练的 ResNet50，修改最后的全连接层"""
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ======================== ResNet50 + CBAM ========================
class CBAMBottleneck(nn.Module):
    """
    带 CBAM 注意力机制的 ResNet Bottleneck 残差块
    在残差连接之前加入 CBAM 模块
    """

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 reduction_ratio=16, spatial_kernel_size=7):
        super(CBAMBottleneck, self).__init__()
        # 兼容 torchvision ResNet._make_layer 的标准参数
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        # 标准 Bottleneck 结构
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=dilation, groups=groups, bias=False, dilation=dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        # CBAM 注意力模块
        self.cbam = CBAMBlock(
            planes * self.expansion,
            reduction_ratio=reduction_ratio,
            spatial_kernel_size=spatial_kernel_size
        )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 在残差连接之前应用 CBAM
        out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def resnet50_cbam(num_classes=4):
    """
    构建 ResNet50 + CBAM 模型：
    基于 ResNet50 预训练权重，将所有 Bottleneck 替换为含 CBAM 的版本
    """
    # 先加载标准 ResNet50 获取预训练权重
    pretrained_model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # 创建新的 ResNet50 骨干，将 Bottleneck 替换为 CBAMBottleneck
    model = models.ResNet(block=CBAMBottleneck, layers=[3, 4, 6, 3])

    # 复制预训练权重（卷积层和 BN 层）
    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    # 只复制匹配的键（跳过 CBAM 新增的参数）
    matched_dict = {}
    for k, v in pretrained_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            matched_dict[k] = v

    model_dict.update(matched_dict)
    model.load_state_dict(model_dict)

    # 修改最后的全连接层
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


# ======================== 模型工厂函数 ========================
def get_model(name, num_classes=None):
    """
    模型工厂函数：根据名称返回对应模型

    Args:
        name: 模型名称 (baseline/resnet18/resnet50/resnet50_cbam)
        num_classes: 分类数，默认使用 config.NUM_CLASSES

    Returns:
        对应的 PyTorch 模型
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES

    if name == 'baseline':
        model = BaselineCNN(num_classes=num_classes)
    elif name == 'resnet18':
        model = resnet18_pretrained(num_classes=num_classes)
    elif name == 'resnet50':
        model = resnet50_pretrained(num_classes=num_classes)
    elif name == 'resnet50_cbam':
        model = resnet50_cbam(num_classes=num_classes)
    else:
        raise ValueError(f'未知模型名称: {name}，可选: {config.MODEL_NAMES}')

    return model


if __name__ == '__main__':
    # 测试所有模型是否能正常创建和前向传播
    from utils import count_parameters

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(2, 3, 224, 224).to(device)

    for name in config.MODEL_NAMES:
        model = get_model(name).to(device)
        output = model(dummy_input)
        params = count_parameters(model)
        print(f'{name}: 输出形状={output.shape}, 可训练参数={params:,}')
        del model
        torch.cuda.empty_cache()
