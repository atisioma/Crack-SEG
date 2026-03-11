import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import yaml
from .data_proc import CrackDataMgr
from .model import CrackFPN
from .loss import CrackDetectionLoss


class CrackTrainTool:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)

        self.epochs = config['Train']['epoch']
        self.device = torch.device(config['Train']['device'])

        self.data_mgr = CrackDataMgr(config['Data']['img_dir'], val_size=config['Train']['val_size'])
        # 二分类，但模型输出1通道（前景概率），num_classes=1
        self.num_classes = 1

        self.train_set = self.data_mgr.get_train_set()
        self.val_set = self.data_mgr.get_val_set()

        self.train_loader = DataLoader(
            self.train_set,
            batch_size=config['Train']['batch_size'],
            shuffle=True,
            num_workers=2,
            drop_last=True
        )
        self.val_loader = DataLoader(
            self.val_set,
            batch_size=config['Train']['batch_size'],
            shuffle=False,
            num_workers=2
        )

        self.model_save_path = config['Train']['save_path']

        self.model = CrackFPN(num_classes=self.num_classes)

        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(config['Train']['learning_rate']),
            weight_decay=1e-4
        )

        # 损失函数 - BCE + Dice
        self.criterion = CrackDetectionLoss(bce_weight=0.5, dice_weight=0.5)

        # 学习率
        self.lr = config['Train']['learning_rate']
        self.lr_step_size = config['Train']['lr_step_size']
        self.lr_gamma = config['Train']['lr_gamma']

        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_step_size, gamma=self.lr_gamma)

        if config['Train']['pretrain_model'] and os.path.exists(config['Train']['pretrain_model']):
            self.model.to(self.device)
            self.model_load_path = config['Train']['pretrain_model']
            self.load_checkpoint()
        else:
            self.model.to(self.device)

    def train_epoch(self, current_epoch):
        self.model.train()
        pbar = tqdm(self.train_loader, desc=f'Train Epoch {current_epoch}')

        running_loss, dice_sum, iou_sum, total_samples = 0.0, 0.0, 0.0, 0

        for inputs, targets in pbar:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # targets: [B, H, W] float32

            self.optimizer.zero_grad()
            outputs = self.model(inputs)  # [B, 1, H, W] logits（模型内部有sigmoid，但BCEWithLogitsLoss会再做一次）

            # 注意：如果模型内部有sigmoid，需要去掉，否则BCEWithLogitsLoss会重复sigmoid
            # 检查model输出是否经过sigmoid，如果是，需要修改model去掉sigmoid

            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            bs = inputs.size(0)
            running_loss += loss.item() * bs
            total_samples += bs

            with torch.no_grad():
                # 计算metrics
                pred_prob = torch.sigmoid(outputs)  # 概率
                pred_mask = (pred_prob > 0.5).float()  # 二值化

                # 确保维度一致
                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)

                inter = (pred_mask * targets).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
                dice = (2. * inter + 1e-5) / (union + 1e-5)
                iou = (inter + 1e-5) / (union - inter + 1e-5)
                dice_sum += dice.mean().item() * bs
                iou_sum += iou.mean().item() * bs

            pbar.set_postfix({
                'Loss': f'{running_loss / total_samples:.4f}',
                'Dice': f'{dice_sum / total_samples:.3f}',
                'IoU': f'{iou_sum / total_samples:.3f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
            })

        return running_loss / total_samples, dice_sum / total_samples, iou_sum / total_samples

    def validate(self):
        self.model.eval()
        pbar = tqdm(self.val_loader, desc='Validate')
        running_loss, dice_sum, iou_sum, total_samples = 0.0, 0.0, 0.0, 0

        with torch.no_grad():
            for inputs, targets in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)  # [B, 1, H, W]
                loss = self.criterion(outputs, targets)

                bs = inputs.size(0)
                running_loss += loss.item() * bs
                total_samples += bs

                pred_prob = torch.sigmoid(outputs)
                pred_mask = (pred_prob > 0.5).float()

                if targets.dim() == 3:
                    targets = targets.unsqueeze(1)

                inter = (pred_mask * targets).sum(dim=(2, 3))
                union = pred_mask.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
                dice = (2. * inter + 1e-5) / (union + 1e-5)
                iou = (inter + 1e-5) / (union - inter + 1e-5)
                dice_sum += dice.mean().item() * bs
                iou_sum += iou.mean().item() * bs

                pbar.set_postfix({
                    'Val Loss': f'{running_loss / total_samples:.4f}',
                    'Val Dice': f'{dice_sum / total_samples:.3f}',
                    'Val IoU': f'{iou_sum / total_samples:.3f}'
                })

    def run(self):
        for epoch in range(self.epochs):
            self.train_epoch(epoch)
            self.scheduler.step()
            if epoch % 10 == 0:
                self.validate()

        self.validate()  # 最后验证一次
        self.save_checkpoint()

    def load_checkpoint(self):
        checkpoint = torch.load(self.model_load_path, map_location='cpu')
        if checkpoint.get('num_classes', 1) != self.num_classes:
            print('[WARN] checkpoints num_class mismatch, skip loading.')
            return
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        saved_epoch = checkpoint.get('epoch', 0)
        init_lr = float(checkpoint.get('init_lr', self.lr))
        step_size = checkpoint.get('init_lr_step_size', self.lr_step_size)
        gamma = checkpoint.get('init_lr_gamma', self.lr_gamma)

        restored_lr = init_lr * (gamma ** (saved_epoch // step_size))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = restored_lr

    def save_checkpoint(self):
        torch.save({
            'num_classes': self.num_classes,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epochs,
            'init_lr': self.lr,
            'init_lr_step_size': self.lr_step_size,
            'init_lr_gamma': self.lr_gamma
        }, self.model_save_path)
