import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiModalModel(nn.Module):
    def __init__(self, num_categories, num_attrs, embed_dim=256, dropout=0.3, pretrained=True):
        super().__init__()
        backbone = models.resnet50(
            weights=models.ResNet50_Weights.DEFAULT if pretrained else None
        )
        self.features = nn.Sequential(*list(backbone.children())[:-2])
        feat_dim = backbone.fc.in_features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.shared = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(dropout)
        )
        self.category_head = nn.Linear(512, num_categories)
        self.attribute_head = nn.Linear(512, num_attrs)

        self.retrieval_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feat_dim, feat_dim//2), nn.BatchNorm1d(feat_dim//2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(feat_dim//2, embed_dim), nn.BatchNorm1d(embed_dim)
        )

    def forward(self, x):
        feats = self.features(x)
        pooled = self.pool(feats)
        shared = self.shared(pooled)
        cat_logits = self.category_head(shared)
        attr_logits = self.attribute_head(shared)
        ret_feat = self.retrieval_proj(pooled)
        ret_embed = F.normalize(ret_feat, dim=1)
        return cat_logits, attr_logits, ret_embed