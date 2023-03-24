import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import InterpolationMode
from torch.autograd import Variable
from torch.utils.data import ConcatDataset

# pretrained models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import resnet152, ResNet152_Weights
from torchvision.models import resnext101_64x4d, ResNeXt101_64X4D_Weights
from torchvision.models import convnext_large, ConvNeXt_Large_Weights

batch_size = 64
epochs = 50
log_file = "log_train.txt"

if (os.path.exists(log_file)):
	os.remove(log_file)

class FashionDataset(Dataset):
    
    # img_dir = "./FashionDataset"
    # img_path_file = "./FashionDataset/split/train.txt" or "./FashionDataset/split/val.txt"
    # label_file = "./FashionDataset/split/train_attr.txt" or "./FashionDataset/split/val_attr.txt"
    
    def __init__(self, img_dir, img_path_file, label_file, transform = None):
        
        self.img_dir = img_dir
        self.img_path_file = img_path_file
        self.label_file = label_file
        self.transform = transform
        
        self.img_paths = []
        self.labels = []
        
        with open(self.img_path_file, "r") as f:
            for line in f:
                self.img_paths.append(os.path.join(self.img_dir, line.strip()))
        
        with open(self.label_file, "r") as f:
            for line in f:
                self.labels.append([int(x) for x in line.strip().split(" ")])
                
    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])
        if self.transform:
            img = self.transform(img)
        return img, self.labels[index]
    
    def __len__(self):
        return len(self.img_paths)

train_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    # transforms.RandomResizedCrop(224,scale=(0.8,1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

img_dir = "./FashionDataset"
train_img_path_file = "./FashionDataset/split/train.txt"
train_label_file = "./FashionDataset/split/train_attr.txt"
val_img_path_file = "./FashionDataset/split/val.txt"
val_label_file = "./FashionDataset/split/val_attr.txt"

training_data = FashionDataset(img_dir, train_img_path_file, train_label_file, train_transform)
val_data = FashionDataset(img_dir, val_img_path_file, val_label_file, train_transform)

dataset = ConcatDataset([training_data, val_data])
train_loader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

# backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
# backbone = resnext101_64x4d(weights = ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
# backbone = resnet152(weights=ResNet152_Weights.IMAGENET1K_V2)
backbone = convnext_large(weights=ConvNeXt_Large_Weights.IMAGENET1K_V1)

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

import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

model = Network(backbone).to(device)

# optimizer
# optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum = 0.9, weight_decay = 0)
optimizer = optim.Adam(model.parameters(), lr = 1e-3)

# scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 2, gamma = 0.8)

criteria_1 = FocalLoss(7, size_average = False).to(device)
criteria_2 = FocalLoss(3, size_average = False).to(device)
criteria_3 = FocalLoss(3, size_average = False).to(device)
criteria_4 = FocalLoss(4, size_average = False).to(device)
criteria_5 = FocalLoss(6, size_average = False).to(device)
criteria_6 = FocalLoss(3, size_average = False).to(device)

l_1 = FocalLoss(7).to(device)
l_2 = FocalLoss(3).to(device)
l_3 = FocalLoss(3).to(device)
l_4 = FocalLoss(4).to(device)
l_5 = FocalLoss(6).to(device)
l_6 = FocalLoss(3).to(device)

train_loss_history = []
train_acc_history = []

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0
    acc_sum = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        for i in range(len(target)):
            target[i] = target[i].to(device)
        optimizer.zero_grad()
        output = model(data)
        
        loss_1 = l_1(output[:, :7], target[0])
        loss_2 = l_2(output[:, 7:10], target[1])
        loss_3 = l_3(output[:, 10:13], target[2])
        loss_4 = l_4(output[:, 13:17], target[3])
        loss_5 = l_5(output[:, 17:23], target[4])
        loss_6 = l_6(output[:, 23:26], target[5])
        
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5 + loss_6

        loss_sum += loss.item()
        
        loss.backward()
        
        optimizer.step()

        pred_1 = output[:, :7].max(1, keepdim=True)[1]
        pred_2 = output[:, 7:10].max(1, keepdim=True)[1]
        pred_3 = output[:, 10:13].max(1, keepdim=True)[1]
        pred_4 = output[:, 13:17].max(1, keepdim=True)[1]
        pred_5 = output[:, 17:23].max(1, keepdim=True)[1]
        pred_6 = output[:, 23:26].max(1, keepdim=True)[1]
        
        correct_1 = pred_1.eq(target[0].view_as(pred_1)).sum().item()
        correct_2 = pred_2.eq(target[1].view_as(pred_2)).sum().item()
        correct_3 = pred_3.eq(target[2].view_as(pred_3)).sum().item()
        correct_4 = pred_4.eq(target[3].view_as(pred_4)).sum().item()
        correct_5 = pred_5.eq(target[4].view_as(pred_5)).sum().item()
        correct_6 = pred_6.eq(target[5].view_as(pred_6)).sum().item()
        
        acc_sum += (correct_1 + correct_2 + correct_3 + correct_4 + correct_5 + correct_6)
        
        if(batch_idx+1)%30 == 0:
            f = open(log_file, "a+")
            f.write('Train Epoch: {} [{}/{} ({:.0f}%)]\t category loss: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}; Total_Loss: {:.6f}\n'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss_1.item(), loss_2.item(), loss_3.item(), loss_4.item(), loss_5.item(), loss_6.item(), loss.item()))
            
            f.write('Train Accuracy: {:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}.\n'.format(correct_1 / batch_size, 
                                                                correct_2 / batch_size,
                                                                correct_3 / batch_size,
                                                                correct_4 / batch_size,
                                                                correct_5 / batch_size,
                                                                correct_6 / batch_size))
            f.close()
    train_acc_history.append( acc_sum / (6 * len(train_loader.dataset)) )
    train_loss_history.append(loss_sum * batch_size / len(train_loader.dataset))
    scheduler.step()

for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    if epoch % 2 == 1 and epoch >= 30:
        torch.save(model.state_dict(), 'model_'+str(epoch)+'.pt')

import matplotlib.pyplot as plt
plt.figure()
plt.title("loss curves")
plt.plot(range(1, epochs + 1), train_loss_history, label="train loss")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig("loss_train.png")

plt.figure()
plt.title("accuracy curves")
plt.plot(range(1, epochs + 1), train_acc_history, label="train acc")
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.legend()
plt.savefig("acc_train.png")

torch.save(model.state_dict(), 'model_final.pt')
