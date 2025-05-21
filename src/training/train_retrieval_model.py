from src.utils.colab_setup import setup_environment
setup_environment()

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from src.models.image_retrieval import RetrievalModel
from dataloaders.deepfashion_retrievals import DeepFashionRetrieval
from utils.triplet_sampling import get_triplet_batch, build_item_index


img_dir = "/content/data/retrieval/img"
anno_path = "/content/data/retrieval/list_eval_partition.txt"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = DeepFashionRetrieval(
    img_dir=img_dir,
    anno_path=anno_path,
    split="train",
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

item_to_indices = build_item_index(train_dataset)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetrievalModel().to(device)

criterion = nn.TripletMarginLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 10
best_loss = float("inf")

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for _ in range(len(train_loader)):
        anchor, positive, negative = get_triplet_batch(train_dataset, item_to_indices, batch_size=32)
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Avg Loss: {avg_loss:.4f}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        torch.save(model.state_dict(), "/content/fashion-multitask-model/retrieval_model_best.pth")
        print("Saved best model.")

torch.save(model.state_dict(), "/content/fashion-multitask-model/retrieval_model_final.pth")
print("Training complete.")




