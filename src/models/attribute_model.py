import torch
import torch.nn as nn
import torchvision.models as models

class AttributeModel(nn.Module):
    def __init__(self, num_categories, num_attributes):
        super(AttributeModel, self).__init__()

        # 1. Loading Resnet18 backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # 2. Removing the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])

        # 3. Shared feature size from ResNet18 output
        feature_dim = 512

        # 4. Classification head
        self.shared_fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # 5. Category classifier
        self.category_head = nn.Linear(256, num_categories)

        # 6. Attribute classifier (multi-label)
        self.attribute_head = nn.Linear(256, num_attributes)

    def forward(self, x):
        # Extracting features from backbone
        x = self.backbone(x)
        x = x.view(x.size(0), -1)  # Flatten [B, 512, 1, 1] → [B, 512]

        # Shared FC layer
        x = self.shared_fc(x)

        # Output branches
        category_logits = self.category_head(x)
        attribute_logits = self.attribute_head(x)

        return category_logits, attribute_logits
