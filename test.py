import os
import numpy as np
from PIL import Image

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable

# pretrained models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

batch_size = 1
model_path = "model_32.pt"

class TestDataset(Dataset):
    
    # img_dir = "./FashionDataset"
    # img_path_file = "./FashionDataset/split/test.txt"
    
    def __init__(self, img_dir, img_path_file, transform = None):
        
        self.img_dir = img_dir
        self.img_path_file = img_path_file
        self.transform = transform
        
        self.img_paths = []
        self.labels = []
        
        with open(self.img_path_file, "r") as f:
            for line in f:
                self.img_paths.append(os.path.join(self.img_dir, line.strip()))
                
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)
        return img
    
    def __len__(self):
        return len(self.img_paths)

test_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_dir = "./FashionDataset"
test_img_path_file = "./FashionDataset/split/test.txt"

if (os.path.exists("prediction.txt")):
        os.remove("prediction.txt")

test_data = TestDataset(img_dir, test_img_path_file, test_transform)
test_loader = DataLoader(test_data, batch_size = batch_size, shuffle = False)

# backbone = resnet50()
# backbone = resnext101_64x4d()
# backbone = resnet152()
backbone = convnext_large()

class Network(nn.Module):
    def __init__(self, backbone):
        super(Network, self).__init__()
        
        self.sequential = nn.Sequential(
            backbone,
            nn.ReLU(),
            nn.Linear(1000, 26)
        )
        
    def forward(self, x):
        return self.sequential(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Network(backbone).to(device)
model.load_state_dict(torch.load(model_path))

def test(model, device, test_loader):
    model.eval()
    print("start test ...")
    f = open("prediction.txt", "a+")
    with torch.no_grad():
        for data in tqdm(test_loader):
            data = data.to(device)
            output = model(data)
            f.write("{} {} {} {} {} {}\n".format(output[:, :7].max(1, keepdim=True)[1].item(),
                                             output[:, 7:10].max(1, keepdim=True)[1].item(),
                                             output[:, 10:13].max(1, keepdim=True)[1].item(),
                                             output[:, 13:17].max(1, keepdim=True)[1].item(),
                                             output[:, 17:23].max(1, keepdim=True)[1].item(),
                                             output[:, 23:26].max(1, keepdim=True)[1].item()))
    f.close()
    print("done!")

test(model, device, test_loader)