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
from sklearn.model_selection import train_test_split
from modeled import UNet,SegmentationModel
from loss import *
from config import *
import torch

torch.cuda.is_available()

from dataload import LoadData,datapre

def getneeddatra(ds,needs=None):
    dsa=torch.zeros(ds.shape)
    for r,sin in enumerate(needs):
        
        nu=datalabel[sin]
        dsa[ds==nu]=r+1
        

    return dsa 

X=sorted(glob.glob(r' '))

outdir=r' '

needs=['Background','Building-flooded','Building-non-flooded','Tree','Grass','Vehicle']


# checkpoint_path = r"K:\Project_engpath\Water_Extraction\SatelliteWaterBodies-master\smpUnet\best_model_mine.pt"


device = 'cuda'


model = SegmentationModel()

model = model.to(device)


model.load_state_dict(torch.load('./checkpoint.pth'))

model.eval()



num=80
ratio=0.5

for file in tqdm(X):
        # file=X[8]
        img=datapre(file).getite()
        logits_mask=model(img.to('cuda', dtype=torch.float32).unsqueeze(0))
        pred_mask=logits_mask.argmax(1)
        
        pred_mask=getneeddatra(pred_mask,needs=needs)
        pre=pred_mask.detach().cpu().numpy().squeeze(0).astype(np.uint8)
        
        
        
        img1=Image.fromarray(pre)
        outfile=outdir+os.path.basename(file)
        img1.save(outfile)
        
        # f, axarr = plt.subplots(1,2) 

        # axarr[0].imshow(np.transpose(img.numpy(), (1,2,0)))
        # axarr[1].imshow(pre,'jet')

