import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score

from fashion_multitask_model.data.dataset import DeepFashionCategories
from fashion_multitask_model.models.category_classification import ClassificationModel

def main():
    # — your CLI parsing here —
    img_dir     = "/path/to/data/img"
    anno_dir    = "/path/to/data/Anno_fine"
    output_dir  = "./checkpoints"
    batch_size  = 64
    lr          = 1e-3
    weight_decay= 1e-4
    epochs      = 10
    patience    = 5
    dropout     = 0.3
    pretrained  = True

    os.makedirs(output_dir, exist_ok=True)

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        transforms.RandomErasing(p=0.5),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    train_ds = DeepFashionCategories(img_dir, anno_dir, 'train', train_tf)
    val_ds   = DeepFashionCategories(img_dir, anno_dir, 'val',   val_tf)

    labels = [lbl for _,lbl in train_ds.samples]
    counts = np.bincount(labels)
    inv_freq = 1.0/(counts+1e-6)
    inv_freq = np.clip(inv_freq, 0.5, 5.0)
    sample_wts = inv_freq[labels]
    sampler = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)
    class_weights = torch.tensor(inv_freq / inv_freq.mean(), device=device)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False,      num_workers=4, pin_memory=True)

    num_categories = int(max(labels)+1)
    model     = ClassificationModel(num_categories, dropout, pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_f1, wait = 0.0, 0
    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch} Train"):
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        print(f"Train loss: {np.mean(train_losses):.4f}")

        model.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for imgs, lbls in tqdm(val_loader, desc=f"Epoch {epoch} Val"):
                imgs = imgs.to(device)
                preds = model(imgs).argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targs.extend(lbls.tolist())
        val_f1 = f1_score(all_targs, all_preds, average='macro')
        print(f"Val F1: {val_f1:.4f}")

        if val_f1 > best_f1 + 1e-4:
            best_f1, wait = val_f1, 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))
            print("  Saved new best model")
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping")
                break
        scheduler.step()

    print("Done. Best Val F1:", best_f1)

if __name__ == "__main__":
    main()
