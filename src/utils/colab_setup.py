import os
import sys

def setup_environment():
    
    from google.colab import drive
    drive.mount('/content/drive')

    
    if not os.path.exists("/content/data"):
        os.system('unzip -q "/content/drive/MyDrive/data.zip" -d /content/')

    
    if not os.path.exists("/content/fashion-multitask-model"):
        os.system("git clone https://github.com/EleneZuroshvili/fashion-multitask-model.git")

    
    os.chdir("/content/fashion-multitask-model")

    
    sys.path.append("./src")

   
    import torch
    print("Environment setup complete. GPU available:", torch.cuda.is_available())
