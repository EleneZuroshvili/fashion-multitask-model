import torch
from models.multitask_classification import MultiTaskModel

def test_multitask_model_output_shapes():
    batch_size = 4
    num_classes = 10
    num_attrs = 8

    model = MultiTaskModel(num_categories=num_classes, num_attrs=num_attrs)
    x = torch.randn(batch_size, 3, 224, 224)

    out_class, out_attr = model(x)

    assert out_class.shape == (batch_size, num_classes), "Category output shape mismatch"
    assert out_attr.shape == (batch_size, num_attrs), "Attribute output shape mismatch"
