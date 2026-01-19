"""
reference from: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/evaluation.py
"""
 
import torch
 
# SR : Segmentation Result
# GT : Ground Truth
 
def get_accuracy(SR,GT):

    corr = torch.sum(SR==GT)
    tensor_size = SR.numel()
    acc = float(corr)/float(tensor_size)
    return acc
 
def get_sensitivity(SR,GT):
    # Sensitivity == Recall

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1)+(GT==1))==2
    FN = ((SR==0)+(GT==1))==2
 
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE
 
def get_specificity(SR,GT):
 
    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0)+(GT==0))==2
    FP = ((SR==1)+(GT==0))==2
 
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP
 
def get_precision(SR,GT):
 
    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1)+(GT==1))==2
    FP = ((SR==1)+(GT==0))==2
 
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
 
    return PC
 
def get_F1(SR,GT):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT)
    PC = get_precision(SR,GT)
 
    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1
 
def get_JS(SR,GT):
    # JS : Jaccard similarity
    
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS
 
def get_DC(SR,GT):
    # DC : Dice Coefficient
    
 
    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC
import numpy as np
def iou(input, target, classes=1):
    """  compute the value of iou
    :param input:  2d array, int, prediction
    :param target: 2d array, int, ground truth
    :param classes: int, the number of class
    :return:
        iou: float, the value of iou
    """
    intersection = np.logical_and(target == classes, input == classes)
    # print(intersection.any())
    union = np.logical_or(target == classes, input == classes)
    iou = np.sum(intersection) / np.sum(union)
    return iou

from sklearn.metrics import confusion_matrix, recall_score, f1_score

needs=['Background','Building-flooded','Building-non-flooded','Tree','Grass','Vehicle']

def CMRSF1(SR,GT):
    cm = confusion_matrix(SR.detach().cpu().numpy(), GT.detach().cpu().numpy())
    recall = recall_score(SR.detach().cpu().numpy(), GT.detach().cpu().numpy(), average='micro')
    f1 = f1_score(SR.detach().cpu().numpy(), GT.detach().cpu().numpy(), average='micro')
    outiou={}
    for r,s in enumerate(needs):
        
        ioua=iou(SR.detach().cpu().numpy(), GT.detach().cpu().numpy(), classes=r)
        outiou[s]=ioua
    meaniou=np.mean(list(outiou.values()))

    
    return cm,recall,f1,outiou,meaniou
