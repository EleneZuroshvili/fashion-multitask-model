import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.models.attribute_model import AttributeModel
from src.dataloaders.deepfashion_attributes import DeepFashionAttributes
from torch.utils.data import Subset

# Data loading
img_dir = "data/attribute pred/img"
anno_dir = "data/attribute pred/Anno_fine"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Training data
train_dataset = DeepFashionAttributes(
    img_dir=img_dir,
    anno_dir=anno_dir,
    split="train",
    transform=transform
)

# Limit to 100 samples for testing
train_dataset = Subset(train_dataset, range(100))

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

num_categories = 50       # DeepFashion has 50 clothing categories
num_attributes = 26       # DeepFashion has 26 attributes

# Model
model = AttributeModel(num_categories=num_categories, num_attributes=num_attributes)

# Loss Functions
criterion_category = nn.CrossEntropyLoss()
criterion_attributes = nn.BCEWithLogitsLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 3

model.train()  # training mode

for epoch in range(num_epochs):
    total_loss = 0

    for images, categories, attributes in train_loader:
        # Forward pass
        category_logits, attribute_logits = model(images)

        # Compute losses
        loss_cat = criterion_category(category_logits, categories)
        loss_attr = criterion_attributes(attribute_logits, attributes.float())
        loss = loss_cat + loss_attr

        # Backward + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

# Saving the model
torch.save(model.state_dict(), "attribute_model.pth")
print("Model saved as attribute_model.pth")

# Evaluation
model.eval()

# Validation data
val_dataset = DeepFashionAttributes(
    img_dir=img_dir,
    anno_dir=anno_dir,
    split="val",
    transform=transform
)

val_loader = DataLoader(val_dataset, batch_size=4, shuffle=True)

# Loading one batch to make predictions
with torch.no_grad():
    for images, true_categories, true_attributes in val_loader:
        pred_cat_logits, pred_attr_logits = model(images)

        # Category prediction: take the index of the highest score
        pred_categories = torch.argmax(pred_cat_logits, dim=1)

        # Attribute prediction: threshold at 0.5
        pred_attributes = (torch.sigmoid(pred_attr_logits) > 0.5).int()

        print("\n🔍 Evaluation on one batch:")
        print("Predicted categories:", pred_categories.tolist())
        print("True categories     :", true_categories.tolist())

        print("Predicted attributes (first sample):", pred_attributes[0][:10].tolist())
        print("True attributes     (first sample):", true_attributes[0][:10].tolist())

        break  # Just check one batch for now
