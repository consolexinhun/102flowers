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
import visdom
import time

train_transform = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                          [0.229,0.224,0.225])
])

test_transform = transforms.Compose([
    lambda x:Image.open(x).convert('RGB'),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                          [0.229,0.224,0.225])
])

class MyDataSet(Dataset):
    def __init__(self, files, mode, labels=None):
        '''
        :param files: 图片的完整路径
        :param labels: 图片标签
        :param mode: 数据集模式 train/test
        '''
        super(MyDataSet, self).__init__()
        self.files =  files
        self.labels = labels
        self.mode = mode

    '''
    如果是训练集，返回样本和标签，如果是测试集，只返回样本
    '''
    def __getitem__(self, item):
        image = self.files[item]
        if self.mode == 'train':
            image = train_transform(image)

            labels = self.labels[item]
            return image, torch.tensor(labels)
        elif self.mode == 'test':
            image = test_transform(image)
            return image

    def __len__(self):
        return len(self.files)

'''
dir 数据集完整路径
mode: 数据集模式 train/test
'''
def split(dir, batch_size, mode):
    images, labels = [], []
    if mode == 'train':
        df = pd.read_csv('./54_data/train.csv')
        if df.shape[0] != len(os.listdir(os.path.join(dir))): # 先判断训练集中的图片和csv中记录的图片数量是不是一样的
            assert "image number not equal train.csv"
        train_files = os.listdir(os.path.join(dir))
        train_files.sort(key=lambda x:int(x[:-4]))
        for i, filename in enumerate(train_files):
            images.append(os.path.join(dir, filename))
            # print(df.loc[df['filename'] == filename].values[0, 1]) # 返回一个二维数组，第一维是几个行，第二维是几列
            labels.append(df.loc[df['filename'] == filename].values[0, 1]) # 把图片对应的标签提取出来
        train_load = DataLoader(MyDataSet(images, 'train', labels), batch_size, shuffle=False)
        return train_load
    elif mode == 'test':
        test_files = os.listdir(os.path.join(dir))
        test_files.sort(key=lambda x:int(x[:-4]))
        for filename in test_files:
            images.append(os.path.join(dir, filename))

        print(images)
        # print(len(images))
        test_load = DataLoader(MyDataSet(images, 'test'), batch_size)
        return test_load

if __name__ == '__main__':
    TrainLoad = split('./54_data/train', 8, 'train')
    '''
    注意显示的时候必须把正则化关了
    '''
    # viz = visdom.Visdom()
    #
    # img, label = next(iter(TrainLoad))
    #
    # viz.images(img, win='batch', opts=dict(title='batch'))
    # viz.text(str(label.numpy()), win='label', opts=dict(title='batch-label'))

    TestLoad = split('./54_data/test', 1, 'test')

    # sample = next(iter(TestLoad))
    # print(sample)
