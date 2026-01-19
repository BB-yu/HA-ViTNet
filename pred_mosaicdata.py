from PIL import Image
import numpy as np
import os
from modeled import UNet,SegmentationModel
import torch
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from skimage import morphology
from skimage import data
from skimage.util import invert

from skimage import morphology, measure

import cv2

from dataload import *

from config import *

Image.MAX_IMAGE_PIXELS = None
device = 'cuda'
model = SegmentationModel()
model = model.to(device)

from PIL import Image
import numpy as np
import torch
from scipy.ndimage import convolve

# Define the color palette (up to 256 colors)
PALETTE = [
    0, 0, 0,        # Class 0: Black
    0, 128, 0,      # Class 1: plant
    192, 192, 192,      # Class 2: road
    238, 232, 170,      # Class 3: field
] + [0, 0, 0] * 252  # Fill the rest of the palette with black


model.load_state_dict(torch.load(outptfile))
model.eval()
def process_crop(crop):
    
    
    
    # crop = torch.tensor(np.array(crop)).permute((2, 0, 1)) / 255.0
    
    
    crop=dataprearray(np.array(crop)).getite()
    
    
    
    crop = crop.to(device, dtype=torch.float32)
    
    
    
    
    
    
    with torch.no_grad():
        logits_mask = model(crop.unsqueeze(0))
        pred_mask = logits_mask.argmax(dim=1).squeeze(0).cpu().numpy()
        
    # print(np.unique(pred_mask))  # Check unique values in the mask
    
    # Convert the mask to 'P' mode image (with a palette)
    pred_mask_img = Image.fromarray(pred_mask.astype(np.uint8), mode='P')
    pred_mask_img.putpalette(PALETTE)

    return pred_mask_img

def crop_and_process(image, crop_size=256, overlap=0.2):
    overlap_size = int(crop_size * overlap)
    stride = crop_size - overlap_size
    width, height = image.size

    # Create a new image in 'P' mode with a palette
    result_image = Image.new('P', (width, height))
    result_image.putpalette(PALETTE)

    for y in trange(0, height, stride):
        for x in range(0, width, stride):
            box = (x, y, x + crop_size, y + crop_size)
            crop = image.crop(box)
            
            if crop.size[0] != crop_size or crop.size[1] != crop_size:
                continue
            
            processed_crop = process_crop(crop)
            result_image.paste(processed_crop, box)
    
    return result_image
def skeletonbest(mask):
    selem = morphology.disk(10)
    closed_image = morphology.closing(mask, selem)
    from skimage.morphology import  remove_small_objects,remove_small_holes
    filled_mask = remove_small_holes(mask,area_threshold=13280)
    closed_image= remove_small_objects(filled_mask,min_size=13280)

    # plt.imshow(closed_image)
    
    dst = np.uint8(closed_image)

    skeleton = np.zeros(dst.shape, np.uint8)
    
    print('skeleton')
    while (True):
        if np.sum(dst) == 0:
            break
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))
        dst = cv2.erode(dst, kernel, None, None, 1)
        open_dst = cv2.morphologyEx(dst, cv2.MORPH_OPEN, kernel)
        result = dst - open_dst
        skeleton = skeleton + result
    
    oit=morphology.dilation(skeleton,selem)
    labeled_skeleton = measure.label(oit)

    regions = measure.regionprops(labeled_skeleton)

    longest_region = max(regions, key=lambda r: r.area)

    longest_skeleton = np.zeros_like(skeleton)
    longest_skeleton[labeled_skeleton == longest_region.label] = 1

    skeleton = morphology.skeletonize(longest_skeleton)
    
    

    def remove_branches(skeleton):

        kernel = np.array([[1, 1, 1],
                        [1, 10, 1],
                        [1, 1, 1]])
        
        while True:

            neighbor_count = convolve(skeleton, kernel, mode='constant', cval=0)
            

            endpoints = (skeleton == 1) & ((neighbor_count == 11) | (neighbor_count == 12))
            

            if not np.any(endpoints):
                break
            

            skeleton = skeleton & ~endpoints
            
        return skeleton

    print(' ')
    cleaned_skeleton = remove_branches(skeleton)
    
    return cleaned_skeleton,closed_image
from scipy.ndimage import distance_transform_edt


import glob

import time 
t1=time.time()
files = glob.glob(' ')




outdir=r'./predict/'
image_path = r'
name1='stitched_output4'
image = Image.open(image_path)

result_image = crop_and_process(image,crop_size=512,  overlap=0.1)
mask=np.array(result_image)==2


import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage import morphology
import matplotlib.pyplot as plt

def remove_small_objects(mask, min_size):

    cleaned_mask = morphology.remove_small_objects(mask, min_size=min_size)
    return cleaned_mask

cleaned_mask = remove_small_objects(mask, min_size=13280)


from scipy import ndimage
def calculate_road_width_simplified(cleaned_mask):  
    

    labeled_image, num_features = ndimage.label(cleaned_mask)  
    sizes = ndimage.sum(cleaned_mask, labeled_image, range(num_features + 1))  
    max_label = np.argmax(sizes)  
    largest_region = (labeled_image == max_label)  


    horizontal_projections = np.sum(largest_region, axis=0)  
    start, end = np.where(horizontal_projections > 0)[0][[0, -1]]  
    vertical_slices = [np.sum(largest_region[:, i]) for i in range(start, end+1)]  
    average_width = np.mean(vertical_slices)  

    if average_width>2200:
        cleaned_mask=np.transpose(cleaned_mask)
        # plt.imshow(cleaned_mask)

        labeled_image, num_features = ndimage.label(cleaned_mask)  
        sizes = ndimage.sum(cleaned_mask, labeled_image, range(num_features + 1))  
        max_label = np.argmax(sizes)  
        largest_region = (labeled_image == max_label)  


        horizontal_projections = np.sum(largest_region, axis=0)  
        start, end = np.where(horizontal_projections > 0)[0][[0, -1]]  
        vertical_slices = [np.sum(largest_region[:, i]) for i in range(start, end+1)]  
        average_width = np.mean(vertical_slices) 
        
    

    return 0, 0, average_width*2


import numpy as np
from scipy import ndimage


def calculate_road_width(cleaned_mask):

    labeled_image, num_features = ndimage.label(cleaned_mask)
    sizes = ndimage.sum(cleaned_mask, labeled_image, range(num_features + 1))
    max_label = np.argmax(sizes)
    largest_region = (labeled_image == max_label)


    skeleton = morphology.skeletonize(largest_region)
    skeleton_coords = np.argwhere(skeleton)

    min_widths = []
    max_widths = []
    for x, y in skeleton_coords:

        distances = np.argwhere(largest_region) - np.array([x, y])
        distances = np.linalg.norm(distances, axis=1)
        min_widths.append(np.min(distances))
        max_widths.append(np.max(distances))

    min_width = np.min(min_widths) if min_widths else 0
    max_width = np.max(max_widths) if max_widths else 0
    average_width = np.mean(min_widths) if min_widths else 0
    
    return min_width, max_width, average_width




min_width, max_width, average_width = calculate_road_width_simplified(cleaned_mask)


H=25
p=2.4
f=24
GSD=H*p/f/3*4
print(f' {average_width*GSD*1e-3}')





result_image.save(outdir+f'{name1}_outpre.png')

result_image.show()
# result_image.show()
t2=time.time()
print(t2-t1)





############################## #################################
cleaned_skeleton,dst=skeletonbest(mask)

road=np.where(dst==1,1,np.nan)



selem = morphology.disk(average_width/20)
cleaned_skeleton1=morphology.dilation(cleaned_skeleton,selem)

h,w=mask.shape
plt.figure(figsize=(w/300,h/300),dpi=300)


plt.imshow(image)
plt.imshow(road,alpha=0.5,cmap='winter')
plt.imshow(np.where(cleaned_skeleton1==1,1,np.nan) )
plt.axis('off')

# plt.savefig(outdir+f'{name1}_中心线.jpg',dpi=300,pad_inches=0,bbox_inches='tight')
plt.tight_layout()
plt.show()





