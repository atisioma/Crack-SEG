from torch.utils.data import Dataset
import cv2
import numpy as np
import torch
import random
from pathlib import Path


class CrackTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, image, mask):

        image = cv2.resize(image, (320, 320), cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (320, 320), cv2.INTER_NEAREST)

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        if self.is_train:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)
                mask = cv2.flip(mask, 1)
            if random.random() > 0.5:
                image = cv2.flip(image, 0)
                mask = cv2.flip(mask, 0)


        image = np.transpose(image, (2, 0, 1))  # CHW

        return torch.from_numpy(image), torch.from_numpy(mask)


class CrackDataSet(Dataset):
    def __init__(self, image_paths, label_paths, transform=None):
        self.transform = transform
        self.image_paths = image_paths
        self.label_paths = label_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label


class CrackDataMgr:
    def __init__(self, img_dir="", val_size=0.2, random_state=42):
        self.img_dir = Path(img_dir)
        self.exts = ['.jpg', '.jpeg', '.png', '.bmp']

        image_paths = []
        label_paths = []

        for ext in self.exts:
            image_paths.extend(list(self.img_dir.glob(f"*{ext}")))

        for img_path in image_paths:
            img_path = Path(img_path)
            label_path = img_path.with_name(img_path.stem + ".png").as_posix().replace("/train_img/", "/train_lab/")
            label_paths.append(label_path)

        self.train_paths, self.val_paths, self.train_labels, self.val_labels = self.train_val_split(
            image_paths, label_paths, val_size, random_state)

    def get_train_set(self) -> CrackDataSet:
        return CrackDataSet(self.train_paths, self.train_labels, CrackTransform(is_train=True))

    def get_val_set(self) -> CrackDataSet:
        return CrackDataSet(self.val_paths, self.val_labels, CrackTransform(is_train=False))

    def train_val_split(self, image_paths, label_paths, val_size, random_state=42):
        random.seed(random_state)

        indices = list(range(len(image_paths)))
        random.shuffle(indices)

        n_val = max(1, int(len(indices) * val_size))
        train_idx = indices[n_val:]
        val_idx = indices[:n_val]

        train_paths = [image_paths[i] for i in train_idx]
        val_paths = [image_paths[i] for i in val_idx]
        train_labels = [label_paths[i] for i in train_idx]
        val_labels = [label_paths[i] for i in val_idx]

        return train_paths, val_paths, train_labels, val_labels