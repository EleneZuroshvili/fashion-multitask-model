from torch.utils.data import Dataset
from PIL import Image
import os
import torch

class DeepFashionAttributes(Dataset):
    def __init__(self, img_dir, anno_dir, split='train', transform=None):
        self.img_dir = img_dir
        self.anno_dir = anno_dir
        self.split = split
        self.transform = transform

        
        with open(os.path.join(anno_dir, split, f"{split}.txt")) as f:
            self.image_names = [line.strip() for line in f]

        
        with open(os.path.join(anno_dir, split, f"{split}_cate.txt")) as f:
            self.category_labels = [int(line.strip()) - 1 for line in f]  # zero-indexed

        
        with open(os.path.join(anno_dir, split, f"{split}_attr.txt")) as f:
            self.attr_labels = [list(map(int, line.strip().split())) for line in f]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx].replace("img/", ""))
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        category = torch.tensor(self.category_labels[idx], dtype=torch.long)
        attributes = torch.tensor(self.attr_labels[idx], dtype=torch.float)

        return image, category, attributes, self.image_names[idx]
