import torch.nn as nn
from torchvision import models

class MultiTaskModel(nn.Module):
    def __init__(self, num_categories, num_attrs, dropout=0.3):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        dim = backbone.fc.in_features

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512),  nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        self.category_head  = nn.Linear(512, num_categories)
        self.attribute_head = nn.Linear(512, num_attrs)

    def forward(self, x):
        x = self.features(x)
        x = self.shared(x)
        return self.category_head(x), self.attribute_head(x)