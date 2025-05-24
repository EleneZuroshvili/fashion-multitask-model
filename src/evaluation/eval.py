import os
import torch
import argparse
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    hamming_loss, recall_score
)
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dataset import (
    DeepFashionCategories,
    DeepFashionAttributes,
    DeepFashionRetrieval,
    DeepFashionMulti
)
from models.category_classification import ClassificationModel
from models.attribute_prediction      import AttributeModel
from models.image_retrieval      import RetrievalModel
from models.multitask_classification      import MultiTaskModel
from models.multitask_final     import MultiModalModel

def get_transform(resize=224):
    return transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

def eval_classification(args):
    tf = get_transform()
    ds = DeepFashionCategories(args.img_dir, args.anno_dir, 'test', transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    n = max(lbl for _,lbl in ds.samples) + 1
    model = ClassificationModel(n, dropout=0.3, pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    preds, targs = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Classif Eval"):
            x = x.to(args.device)
            out = model(x).argmax(1).cpu().tolist()
            preds.extend(out); targs.extend(y.tolist())
    print(f"Acc: {accuracy_score(targs,preds):.4f}  F1: {f1_score(targs,preds,average='macro'):.4f}")
    print(classification_report(targs,preds,digits=4))

def eval_attribute(args):
    tf = get_transform()
    ds = DeepFashionAttributes(args.img_dir, args.anno_dir, 'test', transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = AttributeModel(ds.attr_tensor.shape[1], dropout=0.3, pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    all_preds, all_targs = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Attr Eval"):
            x = x.to(args.device)
            out = torch.sigmoid(model(x)).cpu().numpy() > 0.5
            all_preds.append(out); all_targs.append(y.numpy())
    import numpy as np
    preds = np.vstack(all_preds); targs = np.vstack(all_targs)
    print(f"Hamming Loss: {hamming_loss(targs, preds):.4f}")
    tp = (preds & targs).sum()
    fp = (preds & ~targs).sum()
    fn = (~preds & targs).sum()
    micro_f1 = 2*tp/(2*tp+fp+fn+1e-9)
    print(f"Micro-F1: {micro_f1:.4f}")

def eval_retrieval(args):
    tf = get_transform()
    train_ds = DeepFashionRetrieval(args.img_dir, args.anno_path, 'train', transform=tf)
    query_ds = DeepFashionRetrieval(args.img_dir, args.anno_path, 'query', transform=tf)
    gallery_ds = DeepFashionRetrieval(args.img_dir, args.anno_path, 'gallery', transform=tf)

    items = sorted({item for _,item in train_ds.data})
    idx_map = {it:i for i,it in enumerate(items)}

    ql = DataLoader(query_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)
    gl = DataLoader(gallery_ds,batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = RetrievalModel(embed_dim=args.embed_dim, dropout=0.3, pretrained=False).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    emb_g, ids_g = [], []
    with torch.no_grad():
        for x, iid in tqdm(gl, desc="Building gallery"):
            out = model(x.to(args.device)).cpu()
            emb_g.append(out); ids_g.extend(iid)
    emb_g = torch.cat(emb_g)

    correct = 0; total = 0
    with torch.no_grad():
        for x, iid in tqdm(ql, desc="Query"):
            out = model(x.to(args.device)).cpu()
            sims = out @ emb_g.t()
            top1 = sims.argmax(1)
            for i, qid in enumerate(iid):
                if ids_g[top1[i]] == qid:
                    correct += 1
                total += 1
    print(f"Recall@1: {correct/total:.4f}")

def eval_multitask(args):
    tf = get_transform()
    ds = DeepFashionMulti(args.img_dir, args.anno_dir, 'test', transform=tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    model = MultiTaskModel(
        num_categories = max(c for _,c,_ in ds.samples)+1,
        num_attrs      = len(ds.samples[0][2]),
        dropout=0.3, pretrained=False
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    cat_preds, cat_targs = [], []
    attr_preds, attr_targs = [], []
    with torch.no_grad():
        for x, c, a in tqdm(loader, desc="Multi Eval"):
            out_c, out_a = model(x.to(args.device))
            cat_preds.extend(out_c.argmax(1).cpu().tolist())
            cat_targs.extend(c.tolist())
            pred_a = (torch.sigmoid(out_a)>0.5).cpu().numpy()
            attr_preds.append(pred_a); attr_targs.append(a.numpy())
    import numpy as np
    attr_preds = np.vstack(attr_preds); attr_targs = np.vstack(attr_targs)
    f1_cat = f1_score(cat_targs, cat_preds, average='macro')
    tp = (attr_preds & attr_targs).sum()
    fp = (attr_preds & ~attr_targs).sum()
    fn = (~attr_preds & attr_targs).sum()
    micro_f1 = 2*tp/(2*tp+fp+fn+1e-9)
    print(f"Category Macro-F1 : {f1_cat:.4f}")
    print(f"Attribute micro-F1: {micro_f1:.4f}")

def eval_multimodal(args):
    tfc = get_transform()
    ds_c = DeepFashionAttributes(args.attr_img_dir, args.attr_anno_dir, 'val', transform=tfc)
    loader_c = DataLoader(ds_c, batch_size=args.batch_size, shuffle=False, num_workers=4)
    tf = get_transform()
    query_ds = DeepFashionRetrieval(args.retr_img_dir, args.retr_anno_path, 'query', transform=tf)
    gal_ds   = DeepFashionRetrieval(args.retr_img_dir, args.retr_anno_path, 'gallery', transform=tf)
    loader_q = DataLoader(query_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    loader_g = DataLoader(gal_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = MultiModalModel(
        num_categories = max(c for _,c,_ in DeepFashionAttributes(args.attr_img_dir, args.attr_anno_dir,'train',transform=tfc).attr_tensor.long())+1,
        num_attrs      = ds_c.attr_tensor.shape[1],
        embed_dim      = args.embed_dim,
        dropout=0.3, pretrained=False
    ).to(args.device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    model.eval()

    preds, targs = [], []
    with torch.no_grad():
        for x, c, _ in tqdm(loader_c, desc="Cls Eval"):
            out = model(x.to(args.device))[0].argmax(1).cpu().tolist()
            preds.extend(out); targs.extend(c.tolist())
    f1_cat = f1_score(targs, preds, average='macro')

    emb_g, ids_g = [], []
    with torch.no_grad():
        for x, iid in loader_g:
            out = model(x.to(args.device))[2].cpu()
            emb_g.append(out); ids_g.extend(iid)
    emb_g = torch.cat(emb_g)
    correct = total = 0
    with torch.no_grad():
        for x, iid in loader_q:
            out = model(x.to(args.device))[2].cpu()
            sims = out @ emb_g.t(); top1 = sims.argmax(1)
            for i,q in enumerate(iid):
                if ids_g[top1[i]] == q: correct+=1
                total+=1
    rec1 = correct/total

    print(f"Cat F1: {f1_cat:.4f} | Rec@1: {rec1:.4f}")

def main():
    p = argparse.ArgumentParser(description="Unified Eval for Fashion Models")
    p.add_argument('--task', choices=['classification','attribute','retrieval','multitask','multimodal'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--embed-dim', type=int, default=256)
    # paths
    p.add_argument('--img-dir',    help="For classification/attribute/multitask")
    p.add_argument('--anno-dir',   help="For classification/attribute/multitask")
    p.add_argument('--anno-path',  help="For retrieval list_eval_partition.txt")
    p.add_argument('--attr-img-dir',help="For multimodal")
    p.add_argument('--attr-anno-dir',help="For multimodal")
    p.add_argument('--retr-img-dir',help="For multimodal")
    p.add_argument('--retr-anno-path',help="For multimodal")
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    if args.task == 'classification':
        eval_classification(args)
    elif args.task == 'attribute':
        eval_attribute(args)
    elif args.task == 'retrieval':
        eval_retrieval(args)
    elif args.task == 'multitask':
        eval_multitask(args)
    elif args.task == 'multimodal':
        eval_multimodal(args)

if __name__ == "__main__":
    main()
