from data.dataset import DeepFashionCategories

def test_category_dataset_sample():
    dataset = DeepFashionCategories(
        img_dir='src/tests/fake_data/img',
        anno_dir='src/tests/fake_data/annotations',
        split='test',
        transform=None
    )
    img, label = dataset[0]
    assert img.size == (224, 224), "Image size should be 224x224"
    assert isinstance(label, int), "Label should be an integer"
