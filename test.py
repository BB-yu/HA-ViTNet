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
from loss import *
from evaluate import *
from util import *
from config import *
from modeled import UNet,SegmentationModel
import torch
torch.cuda.is_available()
from dataload import LoadData
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, f1_score,classification_report
import matplotlib.pyplot as plt


def eval_fn(data_loader, model, outjpgfile, num_classes=4):
    model.eval()
    total_diceloss = 0.0
    total_bceloss = 0.0
    test_bar = tqdm(data_loader)

    totalsam = num_corrects = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, masks in test_bar:
            images = images.to(DEVICE, dtype=torch.float32)
            masks = masks.to(DEVICE, dtype=torch.float32)

            logits, bceloss = model(images, masks)
            total_bceloss += bceloss.item()

            predict = logits.argmax(axis=1)
            masks1 = masks.squeeze(axis=1)
            num_correct = torch.eq(predict, masks1).sum().float().item()
            num_corrects += num_correct
            totalsam += np.prod(predict.shape)

            # Collect predictions and targets for evaluation
            all_preds.append(predict.cpu().numpy())
            all_targets.append(masks1.cpu().numpy())

            test_bar.set_description("Test  ACC: %.4f" % (num_corrects / totalsam))

        # Compute metrics
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Flatten arrays for confusion matrix
        all_preds_flat = all_preds.flatten()
        all_targets_flat = all_targets.flatten()

        conf_matrix = confusion_matrix(all_targets_flat, all_preds_flat, labels=np.arange(4))
        
        reports=classification_report(all_targets_flat, all_preds_flat)
        
        print('reports',reports)
        
        
        f1 = f1_score(all_targets_flat, all_preds_flat, average='weighted')
        print('f1',f1)
        # Compute mIoU
        ious = []
        for i in range(num_classes):
            intersection = np.sum((all_targets_flat == i) & (all_preds_flat == i))
            union = np.sum((all_targets_flat == i) | (all_preds_flat == i))
            ious.append(intersection / union if union != 0 else 0)
        mIoU = np.mean(ious)
        print('mIoU',mIoU)
        # Visualization
        if outjpgfile is not None:
            for i in range(1):
                image, mask = next(iter(data_loader))
                sample_num = np.random.randint(0, BATCH_SIZE)
                image = image[sample_num]
                mask = mask[sample_num]
                logits_mask = model(image.unsqueeze(0).to(DEVICE, dtype=torch.float32))
                pred_maska = logits_mask.argmax(axis=1)

                f, axarr = plt.subplots(1, 3)
                axarr[1].imshow(np.squeeze(mask.numpy()), cmap='jet', vmax=3)
                axarr[0].imshow(np.transpose(image[:3].numpy(), (1, 2, 0)))
                axarr[2].imshow(pred_maska.detach().cpu().squeeze(0), cmap='jet', vmax=3)
                plt.savefig(outjpgfile)
                plt.close()

    return total_diceloss / len(data_loader), total_bceloss / len(data_loader), f1, conf_matrix, mIoU

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

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
# %%
train_dataset = LoadData(X_train, y_train)
valid_dataset = LoadData(X_val, y_val)
batch_size = 16
checkpoint_path = "./checkpoint.pth"
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

model.load_state_dict(torch.load(outptfile))

eval_fn(valid_loader,model,outjpgfile=None)



