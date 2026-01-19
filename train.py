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

from dataload import LoadData

def set_seed(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

RANDOM_SEED = 42
set_seed(RANDOM_SEED)


def train_fn(data_loader,model,optimizer):
    model.train()
    total_diceloss=0.0
    total_bceloss=0.0
    num_corrects=0
    totalsam=0
    train_bar=tqdm(data_loader)
    for images ,masks in train_bar:
        images=images.to(DEVICE, dtype=torch.float32)
        masks=masks.to(DEVICE)
        optimizer.zero_grad()
        logits,bceloss=model(images,masks)
        # diceloss.backward(retain_graph=True)
        bceloss.backward()
        optimizer.step()
        # total_diceloss+=diceloss.item()
        total_bceloss+=bceloss.item()
        # break

        train_bar.set_description("Train  loss: %.4f" % (

                total_bceloss/len(data_loader)
            ))

    return total_diceloss/len(data_loader),total_bceloss/len(data_loader)

 # %%
def eval_fn(data_loader,model,outjpgfile):
    model.eval()
    total_diceloss=0.0
    total_bceloss=0.0
    test_bar=tqdm(data_loader)
    
    totalsam=num_corrects=0
    with torch.no_grad():
        for images ,masks in test_bar:
            images=images.to(DEVICE, dtype=torch.float32)
            masks=masks.to(DEVICE, dtype=torch.long)

            logits,bceloss=model(images,masks)
            # total_diceloss+=diceloss.item()
            total_bceloss+=bceloss.item()
            
            # pred_mask=torch.sigmoid(logits)
            # predict = (pred_mask>0.5)*1.0
            predict=logits.argmax(axis=1)
            masks1=masks.squeeze(axis=1)
            num_correct = torch.eq(predict, masks1).sum().float().item()
            num_corrects+= num_correct
            totalsam+= np.prod(predict.shape)
            test_bar.set_description("Test  ACC: %.4f" % (

                    num_corrects / totalsam,
                ))
            

            
        #Visualization
        if outjpgfile is not None:
            for i in range(1):
                image,mask=next(iter(valid_loader))
                
                sample_num=np.random.randint(0,BATCH_SIZE)
                image=image[sample_num]
                mask=mask[sample_num]
                logits_mask=model(image.to('cuda', dtype=torch.float32).unsqueeze(0))
                # da=torch.softmax(logits_mask,axis=1)
                # maska=mask.argmax(axis=1)
                # pred_mask=torch.sigmoid(logits_mask)
                # pred_mask=(pred_mask > ratio)*1.0
                pred_maska=logits_mask.argmax(axis=1)
                ad=np.uint8(np.transpose(image[:3].numpy(), (1,2,0))*255)
                # plt.imshow(ad)
                # plt.show()
                f, axarr = plt.subplots(1,3) 
                axarr[1].imshow(np.squeeze(mask.numpy()), cmap='jet',vmax=7)
                axarr[0].imshow(ad)
                axarr[2].imshow(pred_maska.detach().cpu().squeeze(0), cmap='jet',vmax=7)
                plt.savefig(outjpgfile)
                plt.close()
                # plt.show()
            
    return total_diceloss/len(data_loader),total_bceloss/len(data_loader)


# %%


indir=r'path'
png_pathX = indir+r'/image*.png'
png_pathy =indir+r'/label*.png'
X = sorted(glob.glob(png_pathX))
y = sorted(glob.glob(png_pathy))


truecheck = []
for x1,y1 in zip(X,y):
    # print(x1,y1 )
    if x1.split('_')[-1]==y1.split('_')[-1] and os.path.split(x1)[0]==os.path.split(y1)[0]:
        truecheck.append(1)
if np.unique(truecheck)==1:
    print('-'*30,'数据准确')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = LoadData(X_train[:], y_train[:])
valid_dataset = LoadData(X_val[:], y_val[:],'val')


img, mask = train_dataset[150]

f, axarr = plt.subplots(1,2) 


axarr[1].imshow(np.uint8(np.squeeze(mask.numpy())), cmap='jet')
axarr[0].imshow(np.transpose(img.numpy(), (1,2,0)))


# %%
img.shape

batch_size = BATCH_SIZE


# %% [markdown]
# DataLoader kullanılarak modele girdileri hazırlanıyoruz.

# %%
train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
)

valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
)
device = 'cuda'
model = SegmentationModel()
model = model.to(device)


# optimizer = torch.optim.AdamW(model.parameters(), lr=LR)


optimizer=torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)

scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(  optimizer=optimizer,T_max=EPOCHS,eta_min=0.00001,last_epoch=-1,)
#torch.optim.Adam(model.parameters(),lr=LR)

# %% [markdown]
# **We use the adam optimizer and set up our training loop.**
# 
# **Here we want to save the best model and see our loss at every step...**

# if continue_training:
#     model.load_state_dict(torch.load(loadoutptfile))
#     model.eval()
# %%
best_val_dice_loss=np.inf
best_val_bce_loss=np.inf

for i in range(EPOCHS):
    outfile=basedir+rf"/jpgoutnew/{str(i)}.jpg"
    os.makedirs(os.path.split(outfile)[0],exist_ok=True)
    train_loss = train_fn(train_loader,model,optimizer)
    valid_loss = eval_fn(valid_loader,model,outfile)
    train_dice,train_bce=train_loss
    valid_dice,valid_bce=valid_loss
    scheduler.step(i)
    # 打印当前学习率
    print(optimizer.state_dict()['param_groups'][0]['lr'])

    
    streamaval=f'epch:{str(i+1)},valid_dice:{str(valid_dice / (i+1))},valid_bce:{str(valid_bce / (i+1))}\n'
    streamatrain=f'epch:{str(i+1)},train_dice:{str(train_dice / (i+1))},train_bce:{str(train_bce / (i+1))}\n'
    log(basedir+'/eval.txt',streamaval)
    log(basedir+'/train.txt',streamatrain)
    print(f'Epochs:{i+1}\nTrain_loss --> Dice: {train_dice} BCE: {train_bce} \nValid_loss --> Dice: {valid_dice} BCE: {valid_bce}')
    if valid_dice < best_val_dice_loss or valid_bce < best_val_bce_loss:
        torch.save(model.state_dict(),outptfile)
        print('Model Saved')
        best_val_dice_loss=valid_dice
        best_val_bce_loss=valid_bce

# %%
model.load_state_dict(torch.load(outptfile))
model.eval()
image,mask=valid_dataset[3]

logits_mask=model(image.to(device, dtype=torch.float32).unsqueeze(0))
pred_mask=logits_mask.argmax(axis=1)

# %%
f, axarr = plt.subplots(1,3) 
axarr[1].imshow(np.squeeze(mask.numpy()), cmap='jet')
axarr[0].imshow(np.transpose(image.numpy(), (1,2,0)))
axarr[2].imshow(pred_mask.detach().cpu().squeeze(0), cmap='jet')

