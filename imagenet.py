import numpy as np
from skimage import io
# from skimage import transform
from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
from torchvision.utils import make_grid

class Imagenet_Dataset(Dataset):
    def __init__(self, root_dir, label_file, transform=None):
        '''
        Imagenet训练集的加载
        :param root_dir: 训练集存放的目录
        :param label_file: 标签文件的位置，标签文件中每一行：[文件名 标签]
        :param transform: 是否要在加载的时候进行一些图片变换操作
        '''
        self.root_dir = root_dir
        self.label_file = label_file
        self.transform = transform
        self.size = 0
        self.files_list = []

        if not os.path.isfile(self.label_file):
            print(self.label_file + 'does not exist!')
        file = open(self.label_file)
        for f in file:
            self.files_list.append(f)
            self.size += 1

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # str.split(' ')表示以空格分割字符串
        image_path = self.root_dir + self.files_list[idx].split(' ')[0]
        if not os.path.isfile(image_path):
            print(image_path + 'does not exist!')
            return None
        # image = io.imread(image_path)   # use skitimage
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        label = int(self.files_list[idx].split(' ')[1])
        # 这一块儿需要修改一下
        if self.transform:
            image = self.transform(image)
        return (image, label)