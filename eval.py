import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import glob
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from albumentations import HorizontalFlip, VerticalFlip, Rotate
from tqdm import tqdm
import torch.nn.functional as F
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import seaborn as sns
import pandas as pd
from modeled import *
from loss import *
# from evaluate import *
from util import *
import time
from thop import profile

class LoadData(Dataset):
    def __init__(self, X, y, mode='train'):
        self.X = X
        self.y = y
        self.mode = mode
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        img_path = self.X[idx]
        mask_path = self.y[idx]

        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        image = self.transform(image)
        mask = self.transform(mask)

        # Convert mask to long type for cross entropy loss
        mask = mask.long().squeeze(0)

        return image, mask


indir = r' '
png_pathX = indir + r'image*.png'
png_pathy = indir + r'label*.png'
X = sorted(glob.glob(png_pathX))
y = sorted(glob.glob(png_pathy))

truecheck = []
for x1, y1 in zip(X, y):
    if x1.split('_')[-1] == y1.split('_')[-1] and os.path.split(x1)[0] == os.path.split(y1)[0]:
        truecheck.append(1)
if np.unique(truecheck) == 1:
    print('-' * 30, ' ')

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = LoadData(X_train[:], y_train[:])
valid_dataset = LoadData(X_val[:], y_val[:], 'val')

batch_size = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SegmentationModel()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

# Load model weights
model.load_state_dict(torch.load(outptfile))


def compute_miou(preds, labels, num_classes):

    ious = []
    for cls in range(num_classes):
        pred_cls = (preds == cls).astype(np.float32)
        label_cls = (labels == cls).astype(np.float32)
        
        intersection = np.sum(pred_cls * label_cls)
        union = np.sum(pred_cls + label_cls) - intersection
        
        if union == 0:
            iou = float('nan')
        else:
            iou = intersection / union
        
        ious.append(iou)
    
    return np.nanmean(ious)


def evaluate(model, loader, loss_fn, device, num_classes):
    model.eval()
    epoch_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            masks = torch.squeeze(y, dim=1).long()
            y_pred = model(x)
            loss = loss_fn(y_pred, masks)
            epoch_loss += loss.item()

            _, y_pred_binary = torch.max(y_pred, dim=1)
            all_preds.append(y_pred_binary.cpu().numpy())
            all_labels.append(y.cpu().numpy())

        epoch_loss = epoch_loss / len(loader)
        
    all_preds = np.concatenate(all_preds, axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()


    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))

    report = classification_report(all_labels, all_preds, output_dict=True)
    recall = report['weighted avg']['recall']
    precision = report['weighted avg']['precision']
    

    mpa = np.diag(conf_matrix) / np.sum(conf_matrix, axis=1)
    mpa = np.nanmean(mpa)

    print(classification_report(all_labels, all_preds))
    

    miou = compute_miou(all_preds, all_labels, num_classes)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return epoch_loss, acc, f1, miou, conf_matrix, recall, precision, mpa


def save_results_to_csv(acc, f1, miou, conf_matrix, recall, precision, mpa, output_dir):

    os.makedirs(output_dir, exist_ok=True)
    

    results_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'mIoU', 'Recall', 'Precision', 'MPA'],
        'Value': [acc, f1, miou, recall, precision, mpa]
    })
    
    results_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    

    conf_matrix_df = pd.DataFrame(conf_matrix, index=[f'Class {i}' for i in range(conf_matrix.shape[0])],
                                  columns=[f'Class {i}' for i in range(conf_matrix.shape[1])])
    
    conf_matrix_csv_path = os.path.join(output_dir, 'confusion_matrix.csv')
    conf_matrix_df.to_csv(conf_matrix_csv_path)
    confusion_matrix_decimal = conf_matrix / conf_matrix.sum(axis=0)

    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix_decimal, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f'Evaluation results saved to {results_csv_path}')
    print(f'Confusion matrix saved to {conf_matrix_csv_path}')


def calculate_model_params_and_flops(model, input_shape):
    # Calculate number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    
    # Calculate FLOPs
    dummy_input = torch.randn(input_shape, device=device)
    flops, _ = profile(model, inputs=(dummy_input,))
    
    return total_params, flops


def measure_inference_speed(model, dataloader, device, num_batches=100):
    model.eval()
    total_time = 0
    
    with torch.no_grad():
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device, dtype=torch.float32)
            
            start_time = time.time()
            _ = model(x)
            end_time = time.time()
            
            total_time += end_time - start_time
            
            if i >= num_batches:
                break
    
    avg_inf_time = total_time / num_batches
    inf_speed = 1 / avg_inf_time  # images per second
    
    return inf_speed



num_classes = 6


output_dir = os.path.join(basedir, 'evaluation_results3')


input_shape = (batch_size, 3, 224, 224)
total_params, flops = calculate_model_params_and_flops(model, input_shape)
print(f"Total Parameters: {total_params}")
print(f"FLOPs: {flops}")


inf_speed = measure_inference_speed(model, valid_loader, device)
print(f"Inference Speed: {inf_speed:.2f} images/s")

valid_loss, acc, f1, miou, conf_matrix, recall, precision, mpa = evaluate(model, valid_loader, loss_fn, device, num_classes)

print(f"valid_loss: {valid_loss:.4f}")  
print(f"acc: {acc:.2f}")  
print(f"f1: {f1:.2f}")  
print(f"miou: {miou:.2f}")  
print(f"recall: {recall:.2f}")  
print(f"precision: {precision:.2f}")  
print(f"mpa: {mpa:.2f}")  

# 打印混淆矩阵  
print("conf_matrix:")  
print(conf_matrix)

save_results_to_csv(acc, f1, miou, conf_matrix, recall, precision, mpa, output_dir)



