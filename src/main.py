import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
import cv2
from torch.cuda.amp import autocast, GradScaler

from dataset import ASLDataset
from arch import ResNet

options_path = 'src/config.yml'
with open(options_path, 'r') as f:
  options = yaml.safe_load(f)


torch.backends.cudnn.benchmark = options['train']['benchmark']
torch.backends.cudnn.deterministic = options['train']['deterministic']

def main():
    if not (os.path.exists(options['dataset']['train_path']) and os.path.exists(options['dataset']['test_path'])):
        download_data()
    train_loader, test_loader = make_dataloder()
    
    model = make_model()
    model = model.to(options['device'])  
      
    scaler = GradScaler()
    loss_function = nn.CrossEntropyLoss()
    loss_function = loss_function.to(options['device'])
    optimizer = make_optimizer(model.parameters())
    
    loss_history, acc_history, test_history = train_model(model, optimizer, loss_function, scaler, train_loader, test_loader)
    save_results(model, loss_history, acc_history, test_history)
    

def download_data():
    if options['colab']:
        os.system('pip install kaggle')
    if 'kaggle.json' not in os.listdir():
        raise Exception(f'kaggle.json not found')    
    os.system('mkdir -p ~/.kaggle')
    os.system('cp kaggle.json ~/.kaggle/')
    os.system('chmod 600 ~/.kaggle/kaggle.json')
    os.system('kaggle datasets download "grassknoted/asl-alphabet"')
    os.system('unzip "asl-alphabet.zip"')

def make_dataloder():
    if options['dataset']['transforms']:
        tfs = tv.transforms.Compose([
        tv.transforms.ColorJitter(hue=.50, saturation=.50),
        tv.transforms.RandomRotation(60),
        tv.transforms.Normalize(mean=[0.43,0.44,0.47], std=[0.20,0.20,0.20])
        ])  
    else:
        tfs = None
    train_path = options['dataset']['train_path']
    test_path = options['dataset']['test_path']
    train_dataset = ASLDataset(train_path, transforms=tfs, imsize=128)
    test_dataset = ASLDataset(train_path, test_path, imsize=128)

    batch_size = options['dataset']['batch_size']
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False, drop_last=False)
    return train_loader, test_loader

def make_model():
    params = options['network']
    if  params['arch'] == 'resnet':
        model = ResNet(params['in_nc'], 
                       params['nc'], 
                       params['out_nc'],
                       params['num_blocks'],
                       params['block_type'])
    else:
        raise NotImplementedError(f'architecture {params["arch"]} is not implemented')
    print(count_parameters(model))
    
    if options['train']['add_train']:
        model.load_state_dict(torch.load(options['network']['weights'], 
                                         map_location=options['device']))
    return model

def make_optimizer(model_params):
    params = options['optimizer']
    if params['name'] == 'adam':
        optimizer = torch.optim.Adam(model_params, lr=params['lr'], betas=(params['beta1'], params['beta2']))
    else:
        raise NotImplementedError(f'architecture {params["arch"]} is not implemented')
    return optimizer

def train_model(model, optimizer, loss_function, scaler, train_loader, test_loader):
    loss_history, acc_history, test_history = [], [], []
    params = options['train']
    for epoch in range(params['epochs']):
        model.train()
        loss_val, acc_train, test_acc = 0, 0, 0
        for sample in (pbar := tqdm(train_loader)):
          img, label = sample[0], sample[1]
          img = img.to(options['device'])
          label = label.to(options['device'])
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
        
        loss_history.append(loss_val/len(train_loader))
        acc_history.append(acc_train/len(train_loader))
        print(f'loss: {loss_val/len(train_loader)}')
        print(f'train: {acc_train/len(train_loader)}')
        
        if params['validate']:
            model.eval()
            for sample in test_loader:
                img, label = sample[0], sample[1]
                img = img.to(options['device'])
                label = label.to(options['device'])
                label = F.one_hot(label, 2).float()
                pred = model(img)
                acc_current = accuracy(pred.cpu().float(), label.cpu().float())
                test_acc += acc_current
                test_history.append(test_acc/len(test_loader))
                print(f'test: {test_acc/len(test_loader)}')
        
    return loss_history, acc_history, test_history

def accuracy(pred, label):
    answer = F.softmax(pred.detach()).numpy().argmax(1) == label.numpy().argmax(1)
    return answer.mean()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def save_results(model, loss_history, acc_history, test_history):
    if not os.path.exists(f'{options["name"]}'):
        os.system(f'mkdir {options["name"]}')
    now = datetime.now()
    now = str(now).split('.')[0].replace(' ','_' )
    name = f'{options["network"]["arch"]}_{now}'
    torch.save(model.state_dict(), f'{name}_.pt') 

    with open(f'{options["name"]}/{name}_results.txt', 'w') as f:
        f.writelines(f'loss_history: {str(loss_history)}')
        f.writelines(f'acc_history: {str(acc_history)}')
        f.writelines(f'test_history: {str(test_history)}')
        f.writelines(str(options))

if __name__ == '__main__':
    main()