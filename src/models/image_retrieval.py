import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RetrievalModel(nn.Module):
    def __init__(self, embed_dim=256, dropout=0.3, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        feat_dim = backbone.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim//2), nn.BatchNorm1d(feat_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim//2, embed_dim), nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        out = self.projector(x)
        return F.normalize(out, dim=1)
