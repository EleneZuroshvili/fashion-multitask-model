import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class RetrievalModel(nn.Module):
    def __init__(self, embedding_dim=256):
        super(RetrievalModel, self).__init__()

        # Pretrained Resnet
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(resnet.children())[:-1]  # remove the classification layer
        self.backbone = nn.Sequential(*modules)

        # Projection head to get final embedding
        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(resnet.fc.in_features, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.embedding(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalize embeddings
        return x
