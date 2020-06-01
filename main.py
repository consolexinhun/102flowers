import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader

import os
import csv
import numpy as np
import pandas as pd
from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataload import split

TrainLoad = split('./54_data/train', batch_size=8, mode='train')
TestLoad = split('./54_data/test', batch_size=1, mode='test')

resnet_model = models.resnet18(pretrained=True)
model = nn.Sequential(*list(resnet_model.children())[:-1],
                      Flatten(),
                      nn.Linear(512, 102)).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
criteon = nn.CrossEntropyLoss().to(device)

epochs = 40
for epoch in range(epochs):
    for step, (x, y) in enumerate(TrainLoad):
        model.train()
        x, y = x.to(device), y.to(device)
        logits = model(x)

        loss = criteon(logits, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        if step % 300 == 0:
            print('step:{}, loss:{}'.format(step, loss.item()))

    print('epoch:{}, loss:{}'.format(epoch, loss.item()))

keys, values = [], []
for i in range(len(TestLoad)):
    keys.append(i)

model.eval()
for x in TestLoad:
    x = x.to(device)
    with torch.no_grad():
        out = model(x)
        pred = out.argmax(dim=1)
    # print(pred[0].cpu().numpy())
    values.append(pred[0].cpu().numpy())

with open('key.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(zip(keys, values))










