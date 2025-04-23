import random
from collections import defaultdict
import torch

def build_item_index(dataset):
    item_to_indices = defaultdict(list)
    for idx in range(len(dataset)):
        _, item_id, _ = dataset[idx]
        item_to_indices[item_id].append(idx)
    return item_to_indices

def get_triplet_batch(dataset, item_to_indices, batch_size):
    triplets = []
    for _ in range(batch_size):
        anchor_item_id = random.choice(list(item_to_indices.keys()))
        positive_indices = item_to_indices[anchor_item_id]
        if len(positive_indices) < 2:
            continue  

        anchor_idx, positive_idx = random.sample(positive_indices, 2)

        negative_item_id = random.choice([
            id for id in item_to_indices.keys()
            if id != anchor_item_id and len(item_to_indices[id]) > 0
        ])
        negative_idx = random.choice(item_to_indices[negative_item_id])

        anchor_img, _, _ = dataset[anchor_idx]
        positive_img, _, _ = dataset[positive_idx]
        negative_img, _, _ = dataset[negative_idx]

        triplets.append((anchor_img, positive_img, negative_img))

    if not triplets:
        raise ValueError("Could not sample enough triplets.")

    anchors, positives, negatives = zip(*triplets)
    return (
        torch.stack(anchors),
        torch.stack(positives),
        torch.stack(negatives)
    )
