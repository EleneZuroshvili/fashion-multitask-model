import torch
import torch.nn as nn
from models.attribute_prediction import AttributeModel

def test_training_step_runs():
    # Simulate a batch of 4 images
    x = torch.randn(4, 3, 224, 224)
    # Simulate corresponding binary labels (multi-label, 8 attributes)
    y = torch.randint(0, 2, (4, 8)).float()

    model = AttributeModel(num_attrs=8)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Forward
    output = model(x)
    loss = criterion(output, y)

    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that loss is a float > 0
    assert loss.item() > 0, "Loss should be greater than 0"
