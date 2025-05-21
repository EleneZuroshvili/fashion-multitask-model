import os
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from pytorch_metric_learning import miners, losses as pml_losses

from data.dataset import DeepFashionAttributes, DeepFashionRetrieval
from models.multitask_final import MultiModalModel

def parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--attr-img-dir',   type=str, required=True)
    p.add_argument('--attr-anno-dir',  type=str, required=True)
    p.add_argument('--retr-img-dir',   type=str, required=True)
    p.add_argument('--retr-anno-path', type=str, required=True)
    p.add_argument('--output-dir',     type=str, required=True)
    p.add_argument('--batch-size-cls', type=int,   default=32)
    p.add_argument('--batch-size-ret', type=int,   default=32)
    p.add_argument('--embed-dim',      type=int,   default=256)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight-decay',   type=float, default=1e-4)
    p.add_argument('--epochs',         type=int,   default=20)
    p.add_argument('--pretrained',     action='store_true')
    p.add_argument('--lambda-cls',     type=float, default=0.5, help="Weight for classification/attr loss")
    p.add_argument('--lambda-ret',     type=float, default=1.0, help="Weight for retrieval loss")
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    SEED = 42
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    cls_train_tf = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        transforms.RandomErasing(p=0.5),
    ])
    cls_val_tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    ret_train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
        transforms.RandomErasing(p=0.5),
    ])
    ret_val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])

    attr_ds = DeepFashionAttributes(args.attr_img_dir, args.attr_anno_dir, 'train', transform=cls_train_tf)
    cate_labels = [c for _,c in attr_ds.attr_tensor.long().tolist()]
    counts = np.bincount(cate_labels)
    inv_freq = 1.0/(counts+1e-6)
    inv_freq = np.clip(inv_freq, 0.5, 5.0)
    sample_wts = inv_freq[cate_labels]
    sampler_cls = WeightedRandomSampler(sample_wts, len(sample_wts), replacement=True)
    cls_loader = DataLoader(attr_ds, batch_size=args.batch_size_cls,
                            sampler=sampler_cls, num_workers=4, pin_memory=True)

    retr_ds = DeepFashionRetrieval(args.retr_img_dir, args.retr_anno_path, split="train", transform=ret_train_tf)
    unique_items = sorted({item for _,item in retr_ds.data})
    label_map = {item:idx for idx,item in enumerate(unique_items)}
    retr_labels = [label_map[item] for _,item in retr_ds.data]
    from pytorch_metric_learning.samplers import MPerClassSampler
    retr_loader = DataLoader(
        retr_ds,
        batch_size=args.batch_size_ret,
        sampler=MPerClassSampler(labels=retr_labels, m=4, length_before_new_iter=len(retr_ds)),
        num_workers=4, pin_memory=True
    )

    cls_val_ds  = DeepFashionAttributes(args.attr_img_dir, args.attr_anno_dir, 'val', transform=cls_val_tf)
    cls_val_loader = DataLoader(cls_val_ds, batch_size=args.batch_size_cls,
                                shuffle=False, num_workers=4, pin_memory=True)
    ret_val_ds  = DeepFashionRetrieval(args.retr_img_dir, args.retr_anno_path, split="query", transform=ret_val_tf)
    ret_gallery_ds = DeepFashionRetrieval(args.retr_img_dir, args.retr_anno_path, split="gallery", transform=ret_val_tf)
    from torch.utils.data import ConcatDataset
    ret_query_loader   = DataLoader(ret_val_ds, batch_size=args.batch_size_ret, shuffle=False, num_workers=4, pin_memory=True)
    ret_gallery_loader = DataLoader(ret_gallery_ds, batch_size=args.batch_size_ret, shuffle=False, num_workers=4, pin_memory=True)

    num_categories = int(max(cate_labels))+1
    num_attrs      = attr_ds.attr_tensor.shape[1]
    model = MultiModalModel(num_categories, num_attrs,
                            embed_dim=args.embed_dim,
                            dropout=0.3,
                            pretrained=args.pretrained).to(device)

    pos = attr_ds.attr_tensor.sum(0).to(device)
    neg = len(attr_ds) - pos
    pos_weight = (neg/(pos+1e-6)).clamp(min=1e-3)
    criterion_cls  = nn.CrossEntropyLoss(weight=torch.tensor(inv_freq/inv_freq.mean(), device=device))
    criterion_attr = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    miner     = miners.BatchHardMiner()
    criterion_ret = pml_losses.TripletMarginLoss(margin=0.3)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_score = -1.0
    for epoch in range(1, args.epochs+1):
        model.train()
        cls_iter = iter(cls_loader)
        ret_iter = iter(retr_loader)
        steps = max(len(cls_loader), len(retr_loader))
        total_cls_loss = total_ret_loss = 0.0

        for _ in tqdm(range(steps), desc=f"Epoch {epoch}"):
            optimizer.zero_grad()

            try:
                imgs, attrs = next(cls_iter)
            except StopIteration:
                cls_iter = iter(cls_loader)
                imgs, attrs = next(cls_iter)
            imgs, attrs = imgs.to(device), attrs.to(device)
            cat_logits, attr_logits, _ = model(imgs)
            loss_cls = criterion_cls(cat_logits, torch.tensor([c for _,c in zip(attrs,attrs)], device=device)) \
                       + criterion_attr(attr_logits, attrs)
            total_cls_loss += loss_cls.item()

            try:
                imgs_r, item_ids = next(ret_iter)
            except StopIteration:
                ret_iter = iter(retr_loader)
                imgs_r, item_ids = next(ret_iter)
            imgs_r = imgs_r.to(device)
            labels_r = torch.tensor([label_map[i] for i in item_ids], device=device)
            embeds = model(imgs_r)[2]
            hard_pairs = miner(embeds, labels_r)
            loss_ret = criterion_ret(embeds, labels_r, hard_pairs)
            total_ret_loss += loss_ret.item()

            (args.lambda_cls*loss_cls + args.lambda_ret*loss_ret).backward()
            optimizer.step()

        scheduler.step()

        model.eval()
        all_preds, all_targs = [], []
        with torch.no_grad():
            for imgs, attrs in cls_val_loader:
                imgs = imgs.to(device)
                cat_logits, _, _ = model(imgs)
                preds = cat_logits.argmax(dim=1).cpu().tolist()
                all_preds.extend(preds)
                all_targs.extend(attrs.tolist())
        val_f1 = f1_score(all_targs, all_preds, average='macro')

        gallery_embs, gallery_ids = [], []
        with torch.no_grad():
            for imgs, item_ids in ret_gallery_loader:
                embs = model(imgs.to(device))[2].cpu()
                gallery_embs.append(embs); gallery_ids.extend(item_ids)
            gallery_embs = torch.cat(gallery_embs)

            recall1 = 0
            Q = len(ret_val_ds)
            for imgs, item_ids in ret_query_loader:
                q_embs = model(imgs.to(device))[2].cpu()
                sims = q_embs @ gallery_embs.t()
                top1 = sims.argmax(dim=1)
                for i, q in enumerate(item_ids):
                    if gallery_ids[top1[i]] == q:
                        recall1 += 1
            recall1 /= Q

        combined = 0.5*val_f1 + 0.5*recall1
        print(f"Epoch {epoch} | F1={val_f1:.4f} | R@1={recall1:.4f} | Combined={combined:.4f}")

        if combined > best_score + 1e-4:
            best_score = combined
            save_path = os.path.join(args.output_dir, "best_multimodal.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best multimodal model to {save_path}")

    print("Done. Best Combined Score:", best_score)

if __name__ == "__main__":
    main()
