import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score

from data.dataset import DeepFashionAttributes
from models.attribute_prediction import AttributeModel

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir',     type=str, required=True)
    p.add_argument('--anno-dir',    type=str, required=True)
    p.add_argument('--output-dir',  type=str, required=True)
    p.add_argument('--batch-size',  type=int,   default=64)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight-decay',type=float, default=1e-4)
    p.add_argument('--epochs',      type=int,   default=30)
    p.add_argument('--patience',    type=int,   default=5)
    p.add_argument('--dropout',     type=float, default=0.3)
    p.add_argument('--pretrained',  action='store_true')
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(0.5),
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

    train_ds = DeepFashionAttributes(args.img_dir, args.anno_dir, 'train', transform=train_tf)
    val_ds   = DeepFashionAttributes(args.img_dir, args.anno_dir, 'val',   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)

    attr_tensor = train_ds.attr_tensor.to(device)
    pos = attr_tensor.sum(0)
    neg = len(train_ds) - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1e-3).to(device)

    num_attrs = attr_tensor.shape[1]
    model     = AttributeModel(num_attrs, dropout=args.dropout, pretrained=args.pretrained).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_hamming = float('inf')
    wait = 0
    for epoch in range(1, args.epochs+1):
        model.train()
        for imgs, attrs in tqdm(train_loader, desc=f"Train {epoch:02d}"):
            imgs, attrs = imgs.to(device), attrs.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, attrs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        total_bits = 0
        incorrect  = 0
        tp = fp = fn = 0
        with torch.no_grad():
            for imgs, attrs in tqdm(val_loader, desc=f"Val   {epoch:02d}"):
                imgs, attrs = imgs.to(device), attrs.to(device)
                logits = model(imgs)
                preds = (torch.sigmoid(logits) > 0.5).float()
                incorrect += (preds != attrs).sum().item()
                total_bits += preds.numel()
                tp += ((preds==1)&(attrs==1)).sum().item()
                fp += ((preds==1)&(attrs==0)).sum().item()
                fn += ((preds==0)&(attrs==1)).sum().item()

        hamming = incorrect / total_bits
        micro_f1 = 2*tp/(2*tp + fp + fn + 1e-9)
        print(f"→ Epoch {epoch:02d}: Hamming={hamming:.4f}, Micro-F₁={micro_f1:.4f}")

        if hamming < best_hamming - 1e-6:
            best_hamming = hamming
            wait = 0
            save_path = os.path.join(args.output_dir, "best_attribute_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best attribute model to {save_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break

    print("\nTraining complete. Best Hamming:", best_hamming)

if __name__ == "__main__":
    main()
