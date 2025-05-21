import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

from sklearn.metrics import accuracy_score, f1_score

from data.dataset import DeepFashionMulti
from models.multitask_classification import MultiTaskModel

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

    train_ds = DeepFashionMulti(args.img_dir, args.anno_dir, 'train', transform=train_tf)
    val_ds   = DeepFashionMulti(args.img_dir, args.anno_dir, 'val',   transform=val_tf)

    cate_labels = [c for _,c,_ in train_ds.samples]
    counts      = np.bincount(cate_labels)
    inv_freq    = 1.0/(counts+1e-6)
    inv_freq    = np.clip(inv_freq, 0.5, 5.0)
    sample_wts  = inv_freq[cate_labels]
    sampler     = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)
    class_weights = torch.tensor(inv_freq / inv_freq.mean(), device=device)

    attr_array = torch.tensor([a for _,_,a in train_ds.samples], dtype=torch.float32)
    pos = attr_array.sum(0)
    neg = len(train_ds) - pos
    pos_weight = (neg / (pos + 1e-6)).clamp(min=1e-3).to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False,      num_workers=4, pin_memory=True)

    num_categories = int(max(cate_labels)+1)
    num_attrs      = attr_array.shape[1]
    model          = MultiTaskModel(num_categories, num_attrs,
                                    dropout=args.dropout,
                                    pretrained=args.pretrained).to(device)
    criterion_cate = nn.CrossEntropyLoss(weight=class_weights)
    criterion_attr = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer      = optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    scheduler      = optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=args.epochs, eta_min=1e-5)

    best_combined, wait = -1.0, 0
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss = 0.0
        for imgs, cates, attrs in tqdm(train_loader, desc=f"Train {epoch:02d}"):
            imgs, cates, attrs = imgs.to(device), cates.to(device), attrs.to(device)
            optimizer.zero_grad()
            logit_c, logit_a = model(imgs)
            loss = criterion_cate(logit_c, cates) + criterion_attr(logit_a, attrs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
        avg_train = total_loss / len(train_ds)

        model.eval()
        cat_preds, cat_targs = [], []
        attr_preds_list, attr_targs_list = [], []
        with torch.no_grad():
            for imgs, cates, attrs in tqdm(val_loader, desc=f"Val   {epoch:02d}"):
                imgs = imgs.to(device)
                logit_c, logit_a = model(imgs)
                preds_c = logit_c.argmax(dim=1).cpu().tolist()
                cat_preds.extend(preds_c)
                cat_targs.extend(cates.tolist())
                sig = torch.sigmoid(logit_a).cpu()
                attr_preds_list.append((sig>0.5).float())
                attr_targs_list.append(attrs)

        val_f1      = f1_score(cat_targs, cat_preds, average='macro')
        attr_preds = torch.cat(attr_preds_list)
        attr_targs = torch.cat(attr_targs_list)
        tp = (attr_preds*attr_targs).sum().item()
        fp = (attr_preds*(1-attr_targs)).sum().item()
        fn = ((1-attr_preds)*attr_targs).sum().item()
        micro_f1 = 2*tp/(2*tp + fp + fn + 1e-9)

        combined = 0.5*val_f1 + 0.5*micro_f1
        print(f"Epoch {epoch:02d} | Train Loss={avg_train:.4f}"
              f" | F1_cat={val_f1:.4f} µF1_attr={micro_f1:.4f}"
              f" | Combined={combined:.4f}")

        if combined > best_combined + 1e-4:
            best_combined, wait = combined, 0
            save_path = os.path.join(args.output_dir, "best_multitask_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path}")
        else:
            wait += 1
            if wait >= args.patience:
                print("Early stopping.")
                break
        scheduler.step()

    print("\nTraining complete. Best Combined Score:", best_combined)

if __name__ == "__main__":
    main()
