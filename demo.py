import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os
import random
from sod_model import SODModel

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BEST_PATH = 'best_model_unet.pth'
THRESHOLD = 0.5
IMG_SIZE = 128

# Load model
model = SODModel().to(DEVICE)
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()
print(f"Model loaded on {DEVICE}")


def predict(image_path):
    # Load and preprocess
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}")
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    tensor = torch.tensor(original / 255.0, dtype=torch.float32)
    tensor = tensor.permute(2, 0, 1).unsqueeze(0).to(DEVICE)

    # Inference
    start = time.time()
    with torch.no_grad():
        pred = model(tensor).squeeze().cpu().numpy()
    elapsed = (time.time() - start) * 1000

    # Create outputs
    binary_mask = (pred > THRESHOLD).astype(np.uint8) * 255
    overlay = original.copy()
    overlay[pred > THRESHOLD, 0] = 255
    overlay[pred > THRESHOLD, 1] = int(
        overlay[pred > THRESHOLD, 1].mean() * 0.5)
    overlay[pred > THRESHOLD, 2] = int(
        overlay[pred > THRESHOLD, 2].mean() * 0.5)

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(
        f'Salient Object Detection  |  Inference time: {elapsed:.1f} ms', fontsize=13)

    for ax, data, title in zip(
        axes,
        [original, pred, binary_mask, overlay],
        ['Input image', 'Saliency map', 'Binary mask', 'Overlay']
    ):
        ax.imshow(data, cmap='hot' if title == 'Saliency map' else
                  'gray' if title == 'Binary mask' else None)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('demo_output.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Inference time: {elapsed:.1f} ms")
    print(f"Output saved to demo_output.png")


if __name__ == '__main__':
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        test_dir = 'data/DUTS-TE/DUTS-TE-Image'
        test_img = os.path.join(test_dir, random.choice(os.listdir(test_dir)))
        print(f"Using random image: {test_img}")
        predict(test_img)
