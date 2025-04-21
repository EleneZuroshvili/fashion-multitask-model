# %%
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from src.dataloaders.deepfashion_retrievals import DeepFashionRetrieval
from src.models.retrieval_model import RetrievalModel

img_dir = "/Users/elenezuroshvili/Desktop/Thesis/fashion-multitask-model/data/retrieval/img"
anno_path = "/Users/elenezuroshvili/Desktop/Thesis/fashion-multitask-model/data/retrieval/list_eval_partition.txt"

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

from torch.utils.data import Subset
train_dataset = Subset(train_dataset, range(50))
print(f"🧪 Number of samples: {len(train_dataset)}")

for i in range(3):
    anchor, positive, negative = train_dataset[i]
    print(f"Sample {i}:")
    print("  Anchor type:", type(anchor))
    print("  Positive type:", type(positive))
    print("  Negative type:", type(negative))
    print("  Anchor:", anchor)

batch_size = 4 
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetrievalModel()
model.to(device)

criterion = torch.nn.TripletMarginLoss(margin=1.0)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

import random
from collections import defaultdict

item_to_indices = defaultdict(list)
for idx in range(len(train_dataset)):
    _, item_id, _ = train_dataset[idx]
    item_to_indices[item_id].append(idx)

def get_triplet_batch(dataset, batch_size):
    triplets = []
    for _ in range(batch_size):
        anchor_item_id = random.choice(list(item_to_indices.keys()))
        positive_indices = item_to_indices[anchor_item_id]

        if len(positive_indices) < 2:
            continue  

        anchor_idx, positive_idx = random.sample(positive_indices, 2)

        # Sample a negative from a different item
        negative_item_id = random.choice([id for id in item_to_indices.keys() if id != anchor_item_id])
        negative_idx = random.choice(item_to_indices[negative_item_id])

        anchor_img, _, _ = dataset[anchor_idx]
        positive_img, _, _ = dataset[positive_idx]
        negative_img, _, _ = dataset[negative_idx]


        triplets.append((anchor_img, positive_img, negative_img))

    # Stack into tensors
    anchors, positives, negatives = zip(*triplets)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )

# Training loop 
num_epochs = 1

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    print("🔁 Sampling triplets...")
    for _ in range(len(train_loader)):
        anchor, positive, negative = get_triplet_batch(train_dataset, batch_size=32)

        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        optimizer.zero_grad()

        anchor_embed = model(anchor)
        positive_embed = model(positive)
        negative_embed = model(negative)

        loss = criterion(anchor_embed, positive_embed, negative_embed)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {avg_loss:.4f}")

torch.save(model.state_dict(), "retrieval_model.pth")
print("Model saved as retrieval_model.pth")
# %%
# Evaluation
model.eval()

query_dataset = DeepFashionRetrieval(
    img_dir=img_dir,
    anno_path=anno_path,
    split="query",
    transform=transform
)

with torch.no_grad():
    for i in range(3):
        img, item_id, _ = query_dataset[i]
        img = img.unsqueeze(0).to(device)

        embedding = model(img).cpu().squeeze().numpy()
        print(f"\nQuery Item ID: {item_id}")
        print("Embedding vector (first 10 dims):", embedding[:10])


# %%
