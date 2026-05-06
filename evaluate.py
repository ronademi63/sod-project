import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from data_loader import get_dataloaders
from sod_model import SODModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BEST_PATH = 'best_model_unet.pth'
THRESHOLD = 0.5

train_dl, val_dl, test_dl = get_dataloaders(img_size=128, batch_size=16)

model = SODModel().to(DEVICE)
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()


def compute_metrics(pred_mask, true_mask):
    pred_flat = (pred_mask > THRESHOLD).astype(int).flatten()
    true_flat = true_mask.astype(int).flatten()
    intersection = (pred_flat & true_flat).sum()
    union = (pred_flat | true_flat).sum()
    iou = intersection / (union + 1e-6)
    p = precision_score(true_flat, pred_flat, zero_division=0)
    r = recall_score(true_flat, pred_flat, zero_division=0)
    f1 = f1_score(true_flat, pred_flat, zero_division=0)
    return iou, p, r, f1


# Compute metrics on test set
all_iou, all_p, all_r, all_f1 = [], [], [], []

with torch.no_grad():
    for imgs, masks in test_dl:
        imgs = imgs.to(DEVICE)
        preds = model(imgs).cpu().numpy()
        masks = masks.cpu().numpy()
        for pred, mask in zip(preds, masks):
            iou, p, r, f1 = compute_metrics(pred[0], mask[0])
            all_iou.append(iou)
            all_p.append(p)
            all_r.append(r)
            all_f1.append(f1)

print("=== Test Set Results ===")
print(f"  IoU       : {np.mean(all_iou):.4f}")
print(f"  Precision : {np.mean(all_p):.4f}")
print(f"  Recall    : {np.mean(all_r):.4f}")
print(f"  F1-Score  : {np.mean(all_f1):.4f}")

# Visualize 4 samples
imgs, masks = next(iter(test_dl))
imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

with torch.no_grad():
    preds = model(imgs)

fig, axes = plt.subplots(4, 4, figsize=(14, 14))
for i in range(4):
    img = imgs[i].cpu().permute(1, 2, 0).numpy()
    mask = masks[i].cpu().squeeze().numpy()
    pred = preds[i].cpu().squeeze().numpy()
    overlay = img.copy()
    overlay[pred > THRESHOLD, 0] = 1.0

    for ax, data, title in zip(
        axes[i],
        [img, mask, pred, overlay],
        ['Input', 'Ground truth', 'Predicted', 'Overlay']
    ):
        ax.imshow(data, cmap='gray' if data.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.axis('off')

plt.tight_layout()
plt.savefig('results_visualization.png', dpi=150)
plt.show()
print("Evaluation complete!")
