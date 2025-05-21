import torch.nn as nn
from torchvision import models

class AttributeModel(nn.Module):
    def __init__(self, num_attrs, dropout=0.3, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-1])
        feat_dim = backbone.fc.in_features
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, num_attrs)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x).squeeze()