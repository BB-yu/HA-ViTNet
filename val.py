import os
import glob
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
from modeled import SegmentationModel
from util import LoadDataSingle
import matplotlib.pyplot as plt


infer_dir = r'path'
output_dir = r'path'
os.makedirs(output_dir, exist_ok=True)


device = 'cuda'
model = SegmentationModel()
model.load_state_dict(torch.load(outptfile))
model.to(device)
model.eval()


image_paths = sorted(glob.glob(os.path.join(infer_dir, '*.png')))


tile_size = 512
tiles = {}
for path in image_paths:
    name = os.path.basename(path)
    parts = name.split('.')[0].split('_')
    row, col = int(parts[-2]), int(parts[-1])
    tiles[(row, col)] = path

max_row = max([k[0] for k in tiles.keys()]) + 1
max_col = max([k[1] for k in tiles.keys()]) + 1

full_mask = np.zeros((max_row * tile_size, max_col * tile_size), dtype=np.uint8)


for (row, col), path in tqdm(tiles.items(), desc=' '):
    img = Image.open(path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0
    img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(device)  # [1, 3, H, W]

    with torch.no_grad():
        pred = model(img)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)


    full_mask[row*tile_size:(row+1)*tile_size, col*tile_size:(col+1)*tile_size] = pred


output_mask_path = os.path.join(output_dir, 'full_prediction.png')
Image.fromarray(full_mask).save(output_mask_path)
print(f': {output_mask_path}')


plt.figure(figsize=(12, 12))
plt.imshow(full_mask, cmap='tab20')
plt.axis('off')
plt.title('Full Prediction')
plt.savefig(os.path.join(output_dir, 'full_prediction_vis.png'))
plt.close()
