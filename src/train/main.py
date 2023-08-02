import subprocess
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import cv2
from torch.cuda.amp import autocast, GradScaler
from google.colab import files

from data import ASLDataset
from arch import ResNet

# imsize=128
# tfs True/False 
# batch_size
# device
# change Net
# optim
# params model

def main():
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])

    if 'kaggle.json' not in os.listdir():
        files.upload()
    
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
    subprocess.check_call([sys.executable, 'mkdir', '-p', '~/.kaggle'])
    subprocess.check_call([sys.executable, 'cp', 'kaggle.json', '~/.kaggle/'])
    subprocess.check_call([sys.executable, 'chmod 600 ~/.kaggle/kaggle.json'])
    subprocess.check_call([sys.executable, 'kaggle datasets download', 'grassknoted/asl-alphabet'])
    subprocess.check_call([sys.executable, 'unzip', 'asl-alphabet.zip'])
    
    tfs = tv.transforms.Compose([
        tv.transforms.ColorJitter(hue=.50, saturation=.50),
        # tv.transforms.RandomHorizontalFlip(),
        # tv.transforms.RandomVerticalFlip(),
        tv.transforms.RandomRotation(60),
        # tv.transforms.ToTensor(),
        # tv.transforms.Normalize(mean=[0.43,0.44,0.47],
        #                    std=[0.20,0.20,0.20])
    ])

    train_path = '/content/asl_alphabet_train/asl_alphabet_train'
    test_path = '/content/asl_alphabet_test/asl_alphabet_test'
    train_dataset = ASLDataset.ASLDataset(train_path, transforms=tfs, imsize=128)
    test_dataset = ASLDataset.ASLDataset(train_path, test_path, imsize=128)


    batch_size = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False, drop_last=False)
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    model = ResNet.Net(3, 6, 29, 3, 'bottleneck')

    print(count_parameters(model))
    model = model.to(device)
    scaler = GradScaler()

    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
    
    params = {'epochs': 3,
            'device': 'cuda',
            'use_amp': True}
    loss_history, acc_history, test_history = train_model(model, optimizer, loss_function, 
                                                          scaler, train_loader, 
                                                          test_loader, params)
    
    
def train_model(model, optimizer, loss_function, scaler, train_loader, test_loader, params):
    loss_history, acc_history, test_history = [], [], []
    for epoch in range(params['epochs']):
        model.train()
        loss_val, acc_train, test_acc = 0, 0, 0
        for sample in (pbar := tqdm(train_loader)):
          img, label = sample[0], sample[1]
          img = img.to(params['device'])
          label = label.to(params['device'])
          label = F.one_hot(label, 29).float()
          optimizer.zero_grad()
          with autocast(params['use_amp']):
            pred = model(img)
            loss = loss_function(pred, label)

          scaler.scale(loss).backward()
          loss_item = loss.item()
          loss_val += loss_item

          scaler.step(optimizer)
          scaler.update()

          acc_current = accuracy(pred.cpu().float(), label.cpu().float())
          acc_train += acc_current

          pbar.set_description(f'epoch: {epoch}\tloss: {loss_item:.5f}\taccuracy: {acc_current:.3f}')

        model.eval()
        for sample in test_loader:
            img, label = sample[0], sample[1]
            img = img.to(params['device'])
            label = label.to(params['device'])
            label = F.one_hot(label, 2).float()
            pred = model(img)
            acc_current = accuracy(pred.cpu().float(), label.cpu().float())
            test_acc += acc_current

        test_history.append(test_acc/len(test_loader))
        loss_history.append(loss_val/len(train_loader))
        acc_history.append(acc_train/len(train_loader))
        print(f'loss: {loss_val/len(train_loader)}')
        print(f'train: {acc_train/len(train_loader)}')
        print(f'test: {test_acc/len(test_loader)}')
    return loss_history, acc_history, test_history

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)