import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEModule(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEResNeXtBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 cardinality=32, base_width=4, reduction=16):
        super(SEResNeXtBottleneck, self).__init__()

        width = int(math.floor(planes * (base_width / 64)) * cardinality)

        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)

        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)
        self.se_module = SEModule(planes * 4, reduction=reduction)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se_module(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEResNeXt50Encoder(nn.Module):

    def __init__(self, cardinality=32, base_width=4):
        super(SEResNeXt50Encoder, self).__init__()

        # (conv1)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (layer1-4对应conv2-conv5)
        self.layer1 = self._make_layer(64, 64, 3, stride=1)  # conv2 -> F2
        self.layer2 = self._make_layer(256, 128, 4, stride=2)  # conv3 -> F3
        self.layer3 = self._make_layer(512, 256, 6, stride=2)  # conv4 -> F4
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)  # conv5 -> F5

        self._initialize_weights()

    def _make_layer(self, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * 4, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * 4),
            )

        layers = []
        layers.append(SEResNeXtBottleneck(inplanes, planes, stride, downsample,
                                          cardinality=32, base_width=4))
        for _ in range(1, blocks):
            layers.append(SEResNeXtBottleneck(planes * 4, planes,
                                              cardinality=32, base_width=4))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 输入: (N, 3, 320, 320)
        x = self.conv1(x)  # (N, 64, 160, 160)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # (N, 64, 80, 80)

        f2 = self.layer1(x)  # conv2: (N, 256, 80, 80)  - F2
        f3 = self.layer2(f2)  # conv3: (N, 512, 40, 40)  - F3
        f4 = self.layer3(f3)  # conv4: (N, 1024, 20, 20) - F4
        f5 = self.layer4(f4)  # conv5: (N, 2048, 10, 10) - F5

        return {'F2': f2, 'F3': f3, 'F4': f4, 'F5': f5}


class FPNDecoder(nn.Module):

    def __init__(self, out_channels=256):
        super(FPNDecoder, self).__init__()

        # 1x1卷积用于调整通道数到256
        self.lateral_conv5 = nn.Conv2d(2048, out_channels, kernel_size=1)  # F5 -> P5
        self.lateral_conv4 = nn.Conv2d(1024, out_channels, kernel_size=1)  # F4 -> P4
        self.lateral_conv3 = nn.Conv2d(512, out_channels, kernel_size=1)  # F3 -> P3
        self.lateral_conv2 = nn.Conv2d(256, out_channels, kernel_size=1)  # F2 -> P2

        # 3x3卷积用于消除混叠效应
        self.smooth_conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.smooth_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, features):
        f2, f3, f4, f5 = features['F2'], features['F3'], features['F4'], features['F5']

        # P5
        p5 = self.lateral_conv5(f5)  # (N, 256, 10, 10)

        # P4: 上采样P5 + lateral_conv(F4)
        p5_up = F.interpolate(p5, size=f4.shape[2:], mode='bilinear', align_corners=False)
        p4 = self.lateral_conv4(f4) + p5_up
        p4 = self.smooth_conv4(p4)  # (N, 256, 20, 20)

        # P3: 上采样P4 + lateral_conv(F3)
        p4_up = F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False)
        p3 = self.lateral_conv3(f3) + p4_up
        p3 = self.smooth_conv3(p3)  # (N, 256, 40, 40)

        # P2: 上采样P3 + lateral_conv(F2)
        p3_up = F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False)
        p2 = self.lateral_conv2(f2) + p3_up
        p2 = self.smooth_conv2(p2)  # (N, 256, 80, 80)

        return {'P2': p2, 'P3': p3, 'P4': p4, 'P5': p5}


class AssemblyModule(nn.Module):

    def __init__(self, in_channels=256, out_channels=256, target_size=80):
        super(AssemblyModule, self).__init__()

        self.target_size = target_size

        # P5(10x10) -> 需要3次上采样到80x80
        self.w_ops_5 = nn.ModuleList([
            self._make_w_op(in_channels, in_channels) for _ in range(3)
        ])

        # P4(20x20) -> 需要2次上采样到80x80
        self.w_ops_4 = nn.ModuleList([
            self._make_w_op(in_channels, in_channels) for _ in range(2)
        ])

        # P3(40x40) -> 需要1次上采样到80x80
        self.w_ops_3 = nn.ModuleList([
            self._make_w_op(in_channels, in_channels)
        ])

        # P2(80x80) -> 不需要上采样，但保留3x3卷积用于特征处理
        self.w_op_2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # 最终融合后的卷积
        self.final_conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # 输出层：调整到原始图像大小并生成mask
        self.output_conv = nn.Conv2d(out_channels, 1, kernel_size=1)

    def _make_w_op(self, in_ch, out_ch):
        return nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

    def _apply_w_ops(self, x, w_ops_list):
        for conv in w_ops_list:
            x = conv(x)
            x = F.relu(x, inplace=True)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x

    def forward(self, features):
        p2, p3, p4, p5 = features['P2'], features['P3'], features['P4'], features['P5']

        # H5: P5经过3次W操作 -> (N, 256, 80, 80)
        h5 = self._apply_w_ops(p5, self.w_ops_5)

        # H4: P4经过2次W操作 -> (N, 256, 80, 80)
        h4 = self._apply_w_ops(p4, self.w_ops_4)

        # H3: P3经过1次W操作 -> (N, 256, 80, 80)
        h3 = self._apply_w_ops(p3, self.w_ops_3)

        # H2: P2经过3x3卷积（无需上采样）-> (N, 256, 80, 80)
        h2 = self.w_op_2(p2)
        h2 = F.relu(h2, inplace=True)

        # 特征相加
        fused = h2 + h3 + h4 + h5  # (N, 256, 80, 80)

        # 最终3x3卷积
        fused = self.final_conv(fused)
        fused = F.relu(fused, inplace=True)

        # 上采样到原始图像大小
        output = F.interpolate(fused, scale_factor=4, mode='bilinear', align_corners=False)

        # 生成mask
        pred = self.output_conv(output)  # (N, 1, 320, 320)

        return pred


class CrackFPN(nn.Module):
    def __init__(self, num_classes=1):
        super(CrackFPN, self).__init__()

        self.num_classes = num_classes

        # 编码器：SE-ResNeXt50
        self.encoder = SEResNeXt50Encoder()

        # 解码器：FPN
        self.decoder = FPNDecoder(out_channels=256)

        # 组装模块
        self.assembly = AssemblyModule(in_channels=256, out_channels=256)

    def forward(self, x):
        # 编码器提取特征
        encoder_features = self.encoder(x)

        # FPN解码
        fpn_features = self.decoder(encoder_features)

        # 组装并生成最终pred
        pred = self.assembly(fpn_features)

        return pred
