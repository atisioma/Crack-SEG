import torch
import torch.nn as nn
import torch.nn.functional as F


class CrackDetectionLoss(nn.Module):

    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CrackDetectionLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        # 使用 BCEWithLogitsLoss（自带sigmoid，数值更稳定）
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1.0):
        # pred: [B, 1, H, W] logits 或 [B, H, W]
        # target: [B, 1, H, W] 或 [B, H, W]

        # 先sigmoid得到概率
        pred_prob = torch.sigmoid(pred)

        # 统一维度
        if pred_prob.dim() == 4:
            pred_prob = pred_prob.squeeze(1)  # [B, H, W]
        if target.dim() == 4:
            target = target.squeeze(1)

        pred_flat = pred_prob.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        return 1 - dice

    def forward(self, pred, target):
        # pred: [B, 1, H, W] logits（模型输出，无sigmoid）
        # target: [B, H, W] 或 [B, 1, H, W]，值在{0, 1}

        # 确保target是float且有正确维度
        if target.dim() == 3:
            target = target.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            target = target.float()

        bce = self.bce(pred, target)  # BCEWithLogitsLoss内部做sigmoid
        dice = self.dice_loss(pred, target)

        return self.bce_weight * bce + self.dice_weight * dice