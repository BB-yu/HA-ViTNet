import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
import albumentations as A
from config import *

torch.cuda.is_available()



import torch

torch.cuda.is_available()

    
class datapre(Dataset):
    def __init__(self, images_path):
        super().__init__()
        self.images_path = images_path
        self.transform = A.Compose([
            A.Resize(height,width),
            # A.HorizontalFlip(),
            # A.VerticalFlip(),
            # A.RandomBrightnessContrast(p=0.5),
            # A.GaussNoise(),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        ])
    def getite(self):
        img = Image.open(self.images_path)
        img=np.array(img)
        transformed = self.transform(image=img)
        img = transformed['image']

        img = np.transpose(img, (2, 0, 1))
        img = img/255.0
        img = torch.tensor(img)
        return img

class dataprearray(Dataset):
    def __init__(self, img):
        super().__init__()
        self.img = img
        self.transform = A.Compose([
            # A.Resize(512,512),
            # A.HorizontalFlip(),
            # A.VerticalFlip(),
            # A.RandomBrightnessContrast(p=0.5),
            # A.GaussNoise(),
            # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),

        ])
    def getite(self):

        transformed = self.transform(image=self.img)
        img = transformed['image']

        img = np.transpose(img, (2, 0, 1))
        img = img/255.0
        img = torch.tensor(img)
        return img
class LoadData(Dataset):
    def __init__(self, images_path, masks_path,mode='train'):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.len = len(images_path)
        if mode=='train':
            self.transform = A.Compose([
                
                
                A.Resize(height, width),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.RandomBrightnessContrast(),
                # A.GaussNoise(),
                
        
                # A.Perspective(scale=(0.05, 0.1), p=0.5),

                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),

                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),

            ])
        else:
            self.transform = A.Compose([
                
                
                A.Resize(height, width),
                # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),

            ])


    def __getitem__(self, idx):
        img = Image.open(self.images_path[idx])
        mask = Image.open(self.masks_path[idx])
        # print(self.masks_path[idx])
        
        img,mask=np.array(img),np.array(mask)
        # img1=np.concatenate([img,img],axis=2)
        transformed = self.transform(image=img, mask=mask)
        img = transformed['image']
        mask = transformed['mask']

        
        img = np.transpose(img, (2, 0, 1))
        # img = img/255.0
        img = torch.tensor(img)
        
        mask = np.expand_dims(mask, axis=0)
        # print(np.unique(mask))
        # plt.imshow(mask[0])
        # plt.show()
        
        mask = mask
        mask = torch.tensor(mask).long()

        return img, mask
    
    def __len__(self):
        return self.len
import random
class AddPepperNoise(object):

    def __init__(self, snr, p=0.9):
        assert isinstance(snr, float) or (isinstance(p, float))
        self.snr = snr
        self.p = p

    def __call__(self, img):
        """
        Args:
            img (PIL Image): PIL Image
        Returns:
            PIL Image: PIL image.
        """

        if random.uniform(0, 1) < self.p:

            img_ = np.array(img).copy()

            h, w, c = img_.shape

            signal_pct = self.snr

            noise_pct = (1 - self.snr)

            mask = np.random.choice((0, 1, 2), size=(h, w, 1), p=[signal_pct, noise_pct/2., noise_pct/2.])
            mask = np.repeat(mask, c, axis=2)
            img_[mask == 1] = 255
            img_[mask == 2] = 0

            return Image.fromarray(img_.astype('uint8')).convert('RGB')

        else:
            return img


