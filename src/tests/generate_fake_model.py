import torch
from models.category_classification import ClassificationModel
import os

model = ClassificationModel(num_categories=2)
os.makedirs("src/tests/fake_checkpoints", exist_ok=True)
torch.save(model.state_dict(), "src/tests/fake_checkpoints/fake_model.pth")
