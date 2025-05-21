import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_metric_learning import miners, losses as pml_losses
from pytorch_metric_learning.samplers import MPerClassSampler

from fashion_multitask_model.data.dataset import DeepFashionRetrieval
from fashion_multitask_model.models.image_retrieval import RetrievalModel

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--img-dir',    type=str, required=True)
    p.add_argument('--anno-path',  type=str, required=True,
                   help="Path to list_eval_partition.txt")
    p.add_argument('--output-dir', type=str, required=True)
    p.add_argument('--batch-size', type=int,   default=32)
    p.add_argument('--embed-dim',  type=int,   default=256)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--epochs',     type=int,   default=20)
    p.add_argument('--step-size',  type=int,   default=5)
    p.add_argument('--gamma',      type=float, default=0.5)
    p.add_argument('--m',          type=int,   default=4,
                   help="Number of samples per class in each batch")
    p.add_argument('--pretrained', action='store_true')
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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        transforms.RandomErasing(p=0.5),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    train_ds   = DeepFashionRetrieval(args.img_dir, args.anno_path, split="train", transform=train_tf)
    query_ds   = DeepFashionRetrieval(args.img_dir, args.anno_path, split="query", transform=val_tf)
    gallery_ds = DeepFashionRetrieval(args.img_dir, args.anno_path, split="gallery", transform=val_tf)

    unique_items = sorted({item_id for _, item_id in train_ds.data})
    label_map     = {item: idx for idx, item in enumerate(unique_items)}
    train_labels  = [label_map[item_id] for (_, item_id) in train_ds.data]
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=MPerClassSampler(
            labels=train_labels,
            m=args.m,
            length_before_new_iter=len(train_ds)
        ),
        num_workers=4,
        pin_memory=True
    )

    query_loader   = DataLoader(query_ds,   batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)
    gallery_loader = DataLoader(gallery_ds, batch_size=args.batch_size,
                                shuffle=False, num_workers=4, pin_memory=True)

    model    = RetrievalModel(embed_dim=args.embed_dim,
                              pretrained=args.pretrained).to(device)
    miner    = miners.BatchHardMiner()
    criterion= pml_losses.TripletMarginLoss(margin=0.3)
    optimizer= optim.AdamW(model.parameters(),
                           lr=args.lr, weight_decay=args.weight_decay)
    scheduler= optim.lr_scheduler.StepLR(optimizer,
                                         step_size=args.step_size,
                                         gamma=args.gamma)
    scaler   = torch.cuda.amp.GradScaler()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]")
        for imgs, item_ids in loop:
            imgs = imgs.to(device)
            labels = torch.tensor(
                [label_map[i] for i in item_ids],
                dtype=torch.long, device=device
            )
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                embeddings = model(imgs)
                hard_pairs = miner(embeddings, labels)
                loss = criterion(embeddings, labels, hard_pairs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_ds)
        print(f" → Epoch {epoch} Avg Loss: {avg_loss:.4f}")
        scheduler.step()

    save_path = os.path.join(args.output_dir, "retrieval_best_model.pth")
    torch.save(model.state_dict(), save_path)
    print("Training complete. Model saved to", save_path)

if __name__ == "__main__":
    main()
