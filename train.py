import torch
import os
from data_loader import get_dataloaders
from sod_model import SODModel, combined_loss

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 30
PATIENCE = 5
CKPT_PATH = 'checkpoint_unet.pth'
BEST_PATH = 'best_model_unet.pth'

print(f"Training on: {DEVICE}")

train_dl, val_dl, test_dl = get_dataloaders(img_size=128, batch_size=16)

model = SODModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

start_epoch, best_val_loss, no_improve = 0, float('inf'), 0

# Resume from checkpoint if exists
if os.path.exists(CKPT_PATH):
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    start_epoch = ckpt['epoch'] + 1
    best_val_loss = ckpt['best_val_loss']
    print(f"Resumed from epoch {start_epoch}")

for epoch in range(start_epoch, EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    for imgs, masks in train_dl:
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_dl)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += combined_loss(preds, masks).item()
    val_loss /= len(val_dl)

    print(
        f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ✓ Best model saved (val loss: {val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    # Save checkpoint (bonus)
    torch.save({
        'epoch': epoch,
        'model_state': model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
    }, CKPT_PATH)
    print(f"  ✓ Checkpoint saved at epoch {epoch+1}")

print("Training complete!")
