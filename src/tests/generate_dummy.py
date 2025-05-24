from PIL import Image
import os

os.makedirs("src/tests/fake_data/img", exist_ok=True)
img = Image.new("RGB", (224, 224), color="gray")
img.save("src/tests/fake_data/img/test1.jpg")
