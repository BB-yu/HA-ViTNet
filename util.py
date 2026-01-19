



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
from torch import nn



from config import *
from dataload import *
import numpy as np
import matplotlib.pyplot as plt

def linear_stretch(image, percent):


    stretched_image = np.zeros_like(image, dtype=np.float32)
    

    for i in range(image.shape[0]):

        band = image[i, :, :]
        

        flat_band = band.flatten()
        

        low_percent = np.percentile(flat_band, percent)
        high_percent = np.percentile(flat_band, 100 - percent)
        

        stretched_band = np.clip((band - low_percent) / (high_percent - low_percent), 0, 1)
        

        stretched_image[i, :, :] = stretched_band
    
    return stretched_image
def train_fn(data_loader,model,optimizer):
    model.train()
    total_diceloss=0.0
    total_bceloss=0.0
    num_corrects=0
    totalsam=0
    total_lossn=0
    train_bar=tqdm(data_loader)
    for images ,masks in train_bar:
        
        
        
        images=images.to(DEVICE, dtype=torch.float32)
        masks=masks.to(DEVICE, dtype=torch.float32)

        optimizer.zero_grad()

        logits,diceloss,bceloss=model(images,masks)
        
        
        # diceloss.backward(retain_graph=False)
        
        total_loss=diceloss+bceloss
        total_loss.backward()
        
        optimizer.step()
        # total_diceloss+=diceloss.item()
        # total_bceloss+=bceloss.item()
        
        total_lossn+=total_loss.item()
        # with torch.no_grad():
        #     pred_mask=torch.sigmoid(logits)
        #     predict = (pred_mask>0.5)*1.0
        #     num_correct = torch.eq(predict, masks).sum().float().item()
        #     num_corrects+= num_correct
        #     totalsam+= np.prod(predict.shape)
        #     train_bar.set_description("Train  ACC: %.4f" % (

        #             num_corrects / totalsam,
        #         ))
        # print(total_lossn)
    return total_lossn/len(data_loader)
def calculate_acc(predictions, masks):

    pred_ones = predictions == 1
    mask_ones = masks == 1
    

    correct_ones = torch.logical_and(pred_ones, masks == 1).sum().float().item()


    num_pred_ones = mask_ones.sum().float().item()
    

    if num_pred_ones == 0:
        return 0.0
    

    acc = correct_ones / num_pred_ones
    return acc
# %%
def eval_fn(data_loader,model,outfile):
    model.eval()
    total_diceloss=0.0
    total_bceloss=0.0
    test_bar=tqdm(data_loader)
    total_lossn=0
    totalsam=num_corrects=0
    with torch.no_grad():
        for images ,masks in test_bar:
            images=images.to(DEVICE, dtype=torch.float32)
            masks=masks.to(DEVICE, dtype=torch.float32)

            logits,diceloss,bceloss=model(images,masks)
            # total_diceloss+=diceloss.item()
            # total_bceloss+=bceloss.item()
            total_loss=diceloss+bceloss
            
            
            total_lossn+=total_loss.item()
            
            pred_mask=torch.sigmoid(logits)
            predict = (pred_mask>0.5)*1.0
            num_correct = torch.eq(predict, masks).sum().float().item()
            
            acc=calculate_acc(predict, masks)
            
            
            num_corrects+= num_correct
            totalsam+= np.prod(predict.shape)
            test_bar.set_description("Test  ACC: %.4f" % (

                   num_corrects/totalsam,
                ))
            
            # plt.imshow(masks[0][0].detach().cpu().numpy())
            # plt.show()
            
        #Visualization
        if outfile is not None :
            # for i in range(2):
                # sample_num=6
                sample_num=np.random.randint(0,BATCH_SIZE)

                image,mask=next(iter(data_loader))
                image=image[sample_num]
                mask=mask[sample_num]
                logits_mask=model(image.to('cuda', dtype=torch.float32).unsqueeze(0))
                pred_mask=torch.sigmoid(logits_mask)
                pred_mask=(pred_mask > ratio)*1.0
                
                
                img=linear_stretch(image.numpy()[[2,1,0]], 2)
                img=np.uint8(img*255)
                # img=np.uint8(linear_stretch(img,2))
                
                # plt.imshow(image[-1]*255,'jet')
                # plt.show()
                
                f, axarr = plt.subplots(1,3) 
                axarr[1].imshow(np.squeeze(mask.numpy()), cmap='gray',vmin=0, vmax=1)
                axarr[0].imshow(np.transpose(img, (1,2,0)))
                # if 
                axarr[2].imshow(pred_mask.detach().cpu().squeeze(0)[0].numpy(), cmap='gray',vmin=0, vmax=1)
                plt.tight_layout()
                plt.savefig(outfile,pad_inches=0.01, bbox_inches='tight')
                plt.close()
                # plt.show()
            
    return total_lossn/len(data_loader)
