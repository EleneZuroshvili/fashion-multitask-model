import os
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import Dataset

class DeepFashionCategories(Dataset):

    def __init__(self, img_dir: str, anno_dir: str, split: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        with open(os.path.join(anno_dir, split, f"{split}.txt")) as f:
            names = [l.strip() for l in f]
        with open(os.path.join(anno_dir, split, f"{split}_cate.txt")) as f:
            labs = [int(l.strip()) - 1 for l in f]

        assert len(names) == len(labs), "Mismatch between images and labels"
        self.samples = [(n.replace("img/", ""), lbl) for n, lbl in zip(names, labs)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        path = os.path.join(self.img_dir, fname)
        img  = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class DeepFashionAttributes(Dataset):

    def __init__(self, img_dir: str, anno_dir: str, split: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        with open(os.path.join(anno_dir, split, f"{split}.txt")) as f:
            self.image_names = [l.strip() for l in f]
        with open(os.path.join(anno_dir, split, f"{split}_attr.txt")) as f:
            self.attr_labels = [list(map(int, l.split())) for l in f]

        assert len(self.image_names) == len(self.attr_labels)
        self.attr_tensor = torch.tensor(self.attr_labels, dtype=torch.float)

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        fname = self.image_names[idx].replace("img/", "")
        path  = os.path.join(self.img_dir, fname)
        img   = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        attrs = torch.tensor(self.attr_labels[idx], dtype=torch.float)
        return img, attrs


class DeepFashionRetrieval(Dataset):

    def __init__(self, img_dir: str, anno_path: str, split: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform
        self.data      = []
        with open(anno_path, "r") as f:
            lines = f.readlines()[2:]  
        for line in lines:
            img_name, item_id, subset = line.strip().split()
            if subset == split:
                self.data.append((img_name.replace("img/", ""), item_id))

        self.item_to_indices = defaultdict(list)
        for idx, (_, item_id) in enumerate(self.data):
            self.item_to_indices[item_id].append(idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, item_id = self.data[idx]
        path    = os.path.join(self.img_dir, img_name)
        img     = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, item_id


class DeepFashionMulti(Dataset):

    def __init__(self, img_dir: str, anno_dir: str, split: str, transform=None):
        self.img_dir   = img_dir
        self.transform = transform

        with open(os.path.join(anno_dir, split, f"{split}.txt")) as f:
            names = [l.strip() for l in f]
        with open(os.path.join(anno_dir, split, f"{split}_cate.txt")) as f:
            cates = [int(l.strip()) - 1 for l in f]
        with open(os.path.join(anno_dir, split, f"{split}_attr.txt")) as f:
            attrs = [list(map(int, l.split())) for l in f]

        assert len(names) == len(cates) == len(attrs)
        self.samples = [
            (n.replace("img/", ""), c, a)
            for n, c, a in zip(names, cates, attrs)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, cate, attr = self.samples[idx]
        path  = os.path.join(self.img_dir, fname)
        img   = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cate, torch.tensor(attr, dtype=torch.float)



