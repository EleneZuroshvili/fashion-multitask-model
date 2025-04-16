from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class DeepFashionRetrieval(Dataset):
    def __init__(self, img_dir, anno_path, split="train", transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.data = []

        with open(anno_path, "r") as f:
            lines = f.readlines()[2:]  

        for line in lines:
            img_name, item_id, img_split = line.strip().split()
            if img_split == split:
                self.data.append((img_name, item_id))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, item_id = self.data[idx]
        img_path = os.path.join(self.img_dir, os.path.relpath(img_name, "img"))


        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, item_id, img_name

