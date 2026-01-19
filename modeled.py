
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
from config import *
# from loss import *


class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU()
    
    def forward(self, images):
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        
        return x
class encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = conv(in_channels, out_channels)
        self.pool = nn.MaxPool2d((2,2))

    def forward(self, images):
        x = self.conv(images)
        p = self.pool(x)

        return x, p

# %%
class decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, padding=0)
        self.conv = conv(out_channels * 2, out_channels)

    def forward(self, images, prev):
        x = self.upconv(images)
        x = torch.cat([x, prev], axis=1)
        x = self.conv(x)

        return x

# %% [markdown]
# Burada kafa karıştıran bölüm fonksiyonlar arasında bağlantı olmamasına rağmen fonksiyonların bağlı olması olabilir. Bunu sağlayanın class'ın başlangıcında yazdığımız nn.Module'dür. 
# 
# nn.Module forward fonksiyonunu __init__ ile bağlayıp bir mimarı oluşturuyor...

# %%
class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = encoder(3, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        self.b = conv(512, 1024)

        self.d1 = decoder(1024, 512)
        self.d2 = decoder(512, 256)
        self.d3 = decoder(256, 128)
        self.d4 = decoder(128, 64)

        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)

    def forward(self, images):
        x1, p1 = self.e1(images)
        x2, p2 = self.e2(p1)
        x3, p3 = self.e3(p2)
        x4, p4 = self.e4(p3)

        b = self.b(p4)
        
        d1 = self.d1(b, x4)
        d2 = self.d2(d1, x3)
        d3 = self.d3(d2, x2)
        d4 = self.d4(d3, x1)

        output_mask = self.output(d4)

        return output_mask  
    

class SegmentationModel_or(nn.Module):  

    def __init__(self):  

        super(SegmentationModel_or,self).__init__()  
  

        self.arc = smp.Unet(  
            encoder_name=ENCODER,
            encoder_weights=WEIGHTS,
            in_channels=3,
            classes=3,
            activation=None
        )  

        
        


    def forward(self, images, masks=None):  


        logits = self.arc(images)  


        if masks != None:  

            masks = torch.squeeze(masks, dim = 1).long()
            loss2 = nn.CrossEntropyLoss()(logits, masks)/masks.size(0)

            return logits, loss2  
  

        return logits


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        gap = torch.mean(x, dim=[2, 3])
        gap = self.fc1(gap)
        gap = torch.relu(gap)
        gap = self.fc2(gap)
        gap = self.sigmoid(gap)
        
        gap = gap.view(batch_size, channels, 1, 1)
        return x * gap
from loss import lovasz_softmax

from loss import lovasz_softmax
from OCT2Former import OCT2Former

from networks_2.bra_unet import BRAUnet
from OCT2Former import OCT2Former
from AFT.RWKVdouble import DA_Transformer as ViT_seg
from AFT.RWKVdouble import CONFIGS as CONFIGS_ViT_seg


class SegmentationModel(nn.Module):
    def __init__(self):
        super(SegmentationModel, self).__init__()
        config_vit='R50-ViT-B_16'
        # CONFIGS_ViT_seg[config_vit].n_skip=4
        CONFIGS_ViT_seg[config_vit].patches.grid=(16,16)
        CONFIGS_ViT_seg[config_vit].transformer.num_layers=8
        self.arc = ViT_seg(CONFIGS_ViT_seg[config_vit], img_size=height, num_classes=8)
    def forward(self, images, masks=None):
        # Forward pass through UNet
        logits = self.arc(images)
        if masks is not None:
            masks = torch.squeeze(masks, dim=1).long()
            loss = nn.CrossEntropyLoss()(logits, masks) / masks.size(0)
            loss1=lovasz_softmax(torch.softmax(logits,dim=1), masks, classes='present', per_image=False, ignore=None)
            # loss = FocalLossede()(logits, masks) 
            loss2=0.5*loss+0.5*loss1
            return logits, loss2
        return logits

