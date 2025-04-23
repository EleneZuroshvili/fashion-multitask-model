from src.utils.colab_setup import setup_environment
setup_environment()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from models.attribute_model import AttributeModel
from dataloaders.deepfashion_attributes import DeepFashionAttributes
from sklearn.metrics import f1_score


img_dir = "/content/data/attribute pred/img"
anno_dir = "/content/data/attribute pred/Anno_fine"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


train_dataset = DeepFashionAttributes(
    img_dir=img_dir,
    anno_dir=anno_dir,
    split="train",
    transform=transform
)

val_dataset = DeepFashionAttributes(
    img_dir=img_dir,
    anno_dir=anno_dir,
    split="val",
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset, 
    batch_size=64, 
    shuffle=False)


num_categories = 50       # DeepFashion: 50 categories
num_attributes = 26       # DeepFashion: 26 attributes


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AttributeModel(num_categories=num_categories, num_attributes=num_attributes).to(device)


criterion_category = nn.CrossEntropyLoss()
criterion_attributes = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def evaluate(model, val_loader):
    model.eval()
    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    all_true_attrs = []
    all_pred_attrs = []

    with torch.no_grad():
        for images, categories, attributes in val_loader:
            images = images.to(device)
            categories = categories.to(device)
            attributes = attributes.to(device)

            cat_logits, attr_logits = model(images)

            # --- Category accuracy ---
            preds = torch.argmax(cat_logits, dim=1)
            total_correct += (preds == categories).sum().item()
            total_samples += categories.size(0)

            # --- Attribute F1 ---
            pred_attr = (torch.sigmoid(attr_logits) > 0.5).int().cpu()
            true_attr = attributes.cpu().int()

            all_pred_attrs.append(pred_attr)
            all_true_attrs.append(true_attr)

            # --- Loss ---
            loss_cat = criterion_category(cat_logits, categories)
            loss_attr = criterion_attributes(attr_logits, attributes.float())
            total_loss += (loss_cat + loss_attr).item()

    avg_loss = total_loss / len(val_loader)
    accuracy = total_correct / total_samples

    all_preds = torch.cat(all_pred_attrs, dim=0).numpy()
    all_trues = torch.cat(all_true_attrs, dim=0).numpy()
    f1 = f1_score(all_trues, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, f1


num_epochs = 10
best_val_acc = 0.0
best_val_f1 = 0.0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for images, categories, attributes in train_loader:
        images = images.to(device)
        categories = categories.to(device)
        attributes = attributes.to(device)

        # Forward
        cat_logits, attr_logits = model(images)

        loss_cat = criterion_category(cat_logits, categories)
        loss_attr = criterion_attributes(attr_logits, attributes.float())
        loss = loss_cat + loss_attr

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    val_loss, val_acc, val_f1 = evaluate(model, val_loader)

    print(f"\nEpoch [{epoch+1}/{num_epochs}]")
    print(f"Train Loss: {avg_train_loss:.4f}")
    print(f"Val Loss:   {val_loss:.4f}")
    print(f"Val Acc:    {val_acc*100:.2f}%")
    print(f"Val F1:     {val_f1:.4f}")


    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        save_path = "/content/fashion-multitask-model/attribute_model_best_acc.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Saved best model to {save_path}")

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(model.state_dict(), "/content/fashion-multitask-model/attribute_model_best_f1.pth")
        print(f"Saved best F1 model (F1: {val_f1:.4f})")

print("Training complete.")