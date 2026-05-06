import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SODDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_size=128, augment=False):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.img_size, self.img_size))

        if self.augment:
            if np.random.rand() > 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)
            factor = np.random.uniform(0.7, 1.3)
            img = np.clip(img * factor, 0, 255).astype(np.uint8)

        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        img = torch.tensor(img).permute(2, 0, 1)
        mask = torch.tensor(mask).unsqueeze(0)
        return img, mask


def get_dataloaders(img_size=128, batch_size=16):
    image_dir = 'data/DUTS-TR/DUTS-TR-Image'
    mask_dir = 'data/DUTS-TR/DUTS-TR-Mask'

    images = sorted([os.path.join(image_dir, f)
                    for f in os.listdir(image_dir)])
    masks = sorted([os.path.join(mask_dir,  f) for f in os.listdir(mask_dir)])

    X_train, X_temp, y_train, y_temp = train_test_split(
        images, masks, test_size=0.30, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42)

    train_ds = SODDataset(X_train, y_train, img_size, augment=True)
    val_ds = SODDataset(X_val,   y_val,   img_size, augment=False)
    test_ds = SODDataset(X_test,  y_test,  img_size, augment=False)

    train_dl = DataLoader(train_ds, batch_size=batch_size,
                          shuffle=True,  num_workers=0)
    val_dl = DataLoader(val_ds,   batch_size=batch_size,
                        shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds,  batch_size=batch_size,
                         shuffle=False, num_workers=0)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_dl, val_dl, test_dl
