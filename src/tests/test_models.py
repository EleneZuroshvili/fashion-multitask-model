import torch
from models.attribute_prediction import AttributeModel

def test_attribute_model_output_shape():
    print("Running attribute model shape test...")
    model = AttributeModel(num_attrs=10)  # example with 10 attributes
    dummy_input = torch.randn(4, 3, 224, 224)  # batch of 4 RGB images
    output = model(dummy_input)
    assert output.shape == (4, 10), "Output shape should be [batch_size, num_attrs]"

