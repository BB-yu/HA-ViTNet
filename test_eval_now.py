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
import time, json
from thop import profile
# print(outptfile1)

indir=r'/data/shixinying/Seg/new/crops/*/'
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
    print('-'*30,' ')


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

train_dataset = LoadData(X_train[:], y_train[:])
valid_dataset = LoadData(X_val[:], y_val[:],'val')

batch_size = 4
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


device = 'cuda'
model = SegmentationModel()
model = model.to(device)

loss_fn = nn.CrossEntropyLoss()

# Load model weights
model.load_state_dict(torch.load('path'), strict=False)

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
    all_preds, all_labels = [], []


    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    e2e_times, gpu_times = [], []


    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


    dummy = torch.randn(1, 3, 256, 256).to(device)
    flops, _ = profile(model, inputs=(dummy,), verbose=False)
    flops = flops / 1e9   # GFlops

    reps = 0
    with torch.no_grad():
        for x, y in tqdm(loader, desc='eval'):
            reps += 1
            t0 = time.perf_counter_ns()

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)
            masks = torch.squeeze(y, dim=1).long()

            starter.record()
            y_pred = model(x)
            ender.record()
            torch.cuda.synchronize()
            gpu_ms = starter.elapsed_time(ender)
            gpu_times.append(gpu_ms)


            _, y_pred_binary = torch.max(y_pred, dim=1)
            all_preds.append(y_pred_binary.cpu().numpy())
            all_labels.append(y.cpu().numpy())

            t1 = time.perf_counter_ns()
            e2e_times.append((t1 - t0) / 1e6)   # ms

            loss = loss_fn(y_pred, masks)
            epoch_loss += loss.item()

    epoch_loss /= reps
    all_preds  = np.concatenate(all_preds,  axis=0).flatten()
    all_labels = np.concatenate(all_labels, axis=0).flatten()


    conf_matrix = confusion_matrix(all_labels, all_preds, labels=range(num_classes))
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    recall, precision = report['weighted avg']['recall'], report['weighted avg']['precision']
    mpa = np.nanmean(np.diag(conf_matrix) / conf_matrix.sum(axis=1))
    miou = compute_miou(all_preds, all_labels, num_classes)
    acc  = accuracy_score(all_labels, all_preds)
    f1   = f1_score(all_labels, all_preds, average='weighted')


    speed = {
        'params_M': round(total_params / 1e6, 2),
        'GFLOPs_512x512': round(flops, 2),
        'gpu_infer_ms_mean': round(np.mean(gpu_times), 2),
        'gpu_infer_ms_p99': round(np.percentile(gpu_times, 99), 2),
        'e2e_ms_mean': round(np.mean(e2e_times), 2),
        'e2e_ms_p99': round(np.percentile(e2e_times, 99), 2),
        'fps_gpu_pure': round(1000 / np.mean(gpu_times), 1),
        'fps_e2e': round(1000 / np.mean(e2e_times), 1)
    }


    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'speed_report.json'), 'w') as f:
        json.dump(speed, f, indent=2)
    pd.DataFrame([speed]).to_csv(os.path.join(output_dir, 'speed.csv'), index=False)


    print('------------- Speed & Model Stats -------------')
    print(json.dumps(speed, indent=2))

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
    sns.heatmap(confusion_matrix_decimal, annot=True, fmt='.2f', cmap='Blues', xticklabels=[f'Class {i}' for i in range(conf_matrix.shape[1])],
                yticklabels=[f'Class {i}' for i in range(conf_matrix.shape[0])])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    print(f'Evaluation results saved to {results_csv_path}')
    print(f'Confusion matrix saved to {conf_matrix_csv_path}')


num_classes = 8

output_dir = os.path.join(basedir, 'evaluation_results3')
valid_loss, acc, f1, miou, conf_matrix, recall, precision, mpa = evaluate(model, valid_loader, loss_fn, device, num_classes)

print(f"valid_loss: {valid_loss:.4f}")  
print(f"acc: {acc:.2f}")  
print(f"f1: {f1:.2f}")  
print(f"miou: {miou:.2f}")  
print(f"recall: {recall:.2f}")  
print(f"precision: {precision:.2f}")  
print(f"mpa: {mpa:.2f}")  

print("conf_matrix:")  
print(conf_matrix)

save_results_to_csv(acc, f1, miou, conf_matrix, recall, precision, mpa, output_dir)