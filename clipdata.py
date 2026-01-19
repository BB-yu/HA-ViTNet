import json
import os
from PIL import Image, ImageDraw
import numpy as np
import glob



crop_size =512
overlappr=0.1

files=glob.glob(r' ')

for image_path in files:


    name=image_path.split('\\')[-2]
    image = Image.open(image_path)
    label_image=Image.open(image_path.replace('img','label'))
    # print(np.unique(np.array(label_image)))
    

    # crop_size = 512
    overlap = int(crop_size * overlappr)

    stride = crop_size - overlap
    width, height = image.size

    os.makedirs(f'crops/{name}', exist_ok=True)

    count = 0
    for y in range(0, height, stride):
        for x in range(0, width, stride):

            box = (x, y, x + crop_size, y + crop_size)
            crop = image.crop(box)
            label_tiny= label_image.crop(box)

            if crop.size[0] != crop_size or crop.size[1] != crop_size:
                continue

            # 保存裁剪后的图像
            crop.save(os.path.join(f'crops/{name}', f'image_{count}.png'))
            
            # 保存标签图像
            label_tiny.save(os.path.join(f'crops/{name}', f'label_{count}.png'))
            
            count += 1

    print(f'Total crops created: {count}')


