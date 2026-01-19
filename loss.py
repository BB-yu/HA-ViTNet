
from __future__ import print_function, division
 
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse as ifilterfalse
 
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

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        loss = torch.nn.BCELoss()
        BCE = loss(inputs, targets)
        Dice_BCE = BCE + dice_loss

        return Dice_BCE

class Balanced_CE_loss(torch.nn.Module):
    def __init__(self):
        super(Balanced_CE_loss, self).__init__()

    def forward(self, input, target):
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)
        loss = 0.0
        # version2
        for i in range(input.shape[0]):
            beta = 1-torch.sum(target[i])/target.shape[1]
            x = torch.max(torch.log(input[i]), torch.tensor([-100.0]))
            y = torch.max(torch.log(1-input[i]), torch.tensor([-100.0]))
            l = -(beta*target[i] * x + (1-beta)*(1 - target[i]) * y)
            loss += torch.sum(l)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

class Focal_loss(nn.Module):
    def __init__(self, weight=None, gamma=2):
        super(Focal_loss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = 1e-8
    def forward(self, predict, target):
        if self.weight!=None:
            weights = self.weight.unsqueeze(0).unsqueeze(1).repeat(predict.shape[0], predict.shape[2], 1)
        target_onehot = F.one_hot(target.long(), predict.shape[1]) 
        if self.weight!=None:
            weights = torch.sum(target_onehot * weights, -1)
        input_soft = F.softmax(predict, dim=1)
        probs = torch.sum(input_soft.transpose(2, 1) * target_onehot, -1).clamp(min=0.001, max=0.999)#此处一定要限制范围，否则会出现loss为Nan的现象。
        focal_weight = (1 + self.eps - probs) ** self.gamma
        if self.weight!=None:
            return torch.sum(-torch.log(probs) * weights * focal_weight) / torch.sum(weights)
        else:
            return torch.mean(-torch.log(probs) * focal_weight)





import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossede(nn.Module):
    """
    Focal Loss.

    Args:
        gamma (float): the coefficient of Focal Loss.
        ignore_index (int64): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, gamma=2.0, ignore_index=255):
        super(FocalLossede, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logit, label):
        # Filter out the ignored index in label
        valid_mask = label != self.ignore_index

        # Select valid logits and labels
        logit = logit.permute(0, 2, 3, 1)[valid_mask]  # N,C,H,W -> N,H,W,C -> (N*H*W_valid, C)
        label = label[valid_mask]  # (N*H*W_valid,)

        # Compute log softmax and gather the log probability of the correct class
        logpt = F.log_softmax(logit, dim=-1).gather(dim=-1, index=label.unsqueeze(-1)).squeeze(-1)

        # Compute the focal loss
        pt = logpt.exp()
        loss = -((1 - pt) ** self.gamma) * logpt
        return loss.mean()

# Example usage:
# model = ...
# criterion = FocalLoss(gamma=2.0)
# output = model(input)
# loss = criterion(output, target)




class FocalLoss(nn.Module):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, alpha=[0.15,0.4,0.4,0.05], gamma=2, num_classes=4, size_average=True):
        super(FocalLoss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)
 
        self.gamma = gamma
 
    def forward(self, preds, labels):
        preds = preds.view(preds.size(0),preds.size(1), -1)
        alpha_array=torch.tensor(labels.clone().detach(),dtype=torch.float32)
        preds_softmax = F.softmax(preds,dim=1)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(labels.size(0), 1,-1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(labels.size(0), 1,-1))
 
        for i in range(len(self.alpha)):
            alpha_array[alpha_array==i]=self.alpha[i] 
            
        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft)
        loss = torch.mul(alpha_array.view(alpha_array.size(0),-1), loss.view(loss.size(0),-1))
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        # self.alpha=[0.15,0.4,0.4,0.05]
        return loss

class FocalLoss_q(nn.Module):
    def __init__(self, gamma=2):
        super(FocalLoss_q, self).__init__()
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs = inputs.sigmoid()
        loss = -(1 - inputs) ** self.gamma * targets * torch.log(inputs) - inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        loss = loss.mean()
        return loss

# criterion = FocalLoss()

"""
Lovasz-Softmax and Jaccard hinge loss in PyTorch
Maxim Berman 2018 ESAT-PSI KU Leuven (MIT License)
"""
 

 

def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    # print("p = ", p)
    # print("gt_sorted = ", gt_sorted)
    gts = gt_sorted.sum()

    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard
 
 
def iou_binary(preds, labels, EMPTY=1., ignore=None, per_image=True):
    """
    IoU for foreground class
    binary: 1 foreground, 0 background
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        intersection = ((label == 1) & (pred == 1)).sum()
        union = ((label == 1) | ((pred == 1) & (label != ignore))).sum()
        if not union:
            iou = EMPTY
        else:
            iou = float(intersection) / float(union)
        ious.append(iou)
    iou = mean(ious)    # mean accross images if per_image
    return 100 * iou
 
 
def iou(preds, labels, C, EMPTY=1., ignore=None, per_image=False):
    """
    Array of IoU for each (non ignored) class
    """
    if not per_image:
        preds, labels = (preds,), (labels,)
    ious = []
    for pred, label in zip(preds, labels):
        iou = []    
        for i in range(C):
            if i != ignore: # The ignored label is sometimes among predicted classes (ENet - CityScapes)
                intersection = ((label == i) & (pred == i)).sum()
                union = ((label == i) | ((pred == i) & (label != ignore))).sum()
                if not union:
                    iou.append(EMPTY)
                else:
                    iou.append(float(intersection) / float(union))
        ious.append(iou)
    ious = [mean(iou) for iou in zip(*ious)] # mean accross images if per_image
    return 100 * np.array(ious)
 
 
# --------------------------- BINARY LOSSES ---------------------------
 
 
def lovasz_hinge(logits, labels, per_image=True, ignore=None):
    """
    Binary Lovasz hinge loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      per_image: compute the loss per image instead of per batch
      ignore: void class id
    """
    if per_image:
        loss = mean(lovasz_hinge_flat(*flatten_binary_scores(log.unsqueeze(0), lab.unsqueeze(0), ignore))
                          for log, lab in zip(logits, labels))
    else:
        loss = lovasz_hinge_flat(*flatten_binary_scores(logits, labels, ignore))
    return loss
 
 
def lovasz_hinge_flat(logits, labels):
    """
    Binary Lovasz hinge loss
      logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
      labels: [P] Tensor, binary ground truth labels (0 or 1)
      ignore: label to ignore
    """
    if len(labels) == 0:
        # only void pixels, the gradients should be 0
        return logits.sum() * 0.
    signs = 2. * labels.float() - 1.
    errors = (1. - logits * Variable(signs))
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), Variable(grad))
    return loss
 
 
def flatten_binary_scores(scores, labels, ignore=None):
    """
    Flattens predictions in the batch (binary case)
    Remove labels equal to 'ignore'
    """
    scores = scores.view(-1)
    labels = labels.view(-1)
    if ignore is None:
        return scores, labels
    valid = (labels != ignore)
    vscores = scores[valid]
    vlabels = labels[valid]
    return vscores, vlabels
 
 
class StableBCELoss(torch.nn.modules.Module):
    def __init__(self):
         super(StableBCELoss, self).__init__()
    def forward(self, input, target):
         neg_abs = - input.abs()
         loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
         return loss.mean()
 
 
def binary_xloss(logits, labels, ignore=None):
    """
    Binary Cross entropy loss
      logits: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
      labels: [B, H, W] Tensor, binary ground truth masks (0 or 1)
      ignore: void class id
    """
    logits, labels = flatten_binary_scores(logits, labels, ignore)
    loss = StableBCELoss()(logits, Variable(labels.float()))
    return loss
 
 
# --------------------------- MULTICLASS LOSSES ---------------------------
 
 
def lovasz_softmax(probas, labels, classes='present', per_image=False, ignore=None):
    """
    Multi-class Lovasz-Softmax loss
      probas: [B, C, H, W] Variable, class probabilities at each prediction (between 0 and 1).
              Interpreted as binary (sigmoid) output with outputs of size [B, H, W].
      labels: [B, H, W] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """
    # print("probas.shape = ", probas.shape)
    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore), classes=classes)
                          for prob, lab in zip(probas, labels))
    else:
    	  #lovasz_softmax_flat的输入就是probas 【262144 2】  labels【262144】
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore), classes=classes)
    return loss
 
 

def lovasz_softmax_flat(probas, labels, classes='present'):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
    """

    if probas.numel() == 0:
        # only void pixels, the gradients should be 0
        return probas * 0.
    C = probas.size(1)
    losses = []

    class_to_sum = list(range(C)) if classes in ['all', 'present'] else classes
    for c in class_to_sum:
        fg = (labels == c).float()
        if (classes is 'present' and fg.sum() == 0):
            continue
        if C == 1:
            if len(classes) > 1:
                raise ValueError('Sigmoid output possible only with 1 class')
            class_pred = probas[:, 0]
        else:
            class_pred = probas[:, c]

        errors = (Variable(fg) - class_pred).abs()

        errors_sorted, perm = torch.sort(errors, 0, descending=True)
 
        perm = perm.data

        fg_sorted = fg[perm]
        
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)
 
 
def flatten_probas(probas, labels, ignore=None):
    """
    Flattens predictions in the batch
    """

    if probas.dim() == 3:
        B, H, W = probas.size()
        probas = probas.view(B, 1, H, W)
    B, C, H, W = probas.size()

    probas = probas.permute(0, 2, 3, 1).contiguous().view(-1, C)  # B * H * W, C = P, C
    #
    labels = labels.view(-1)

    if ignore is None:
        return probas, labels
    valid = (labels != ignore)
    vprobas = probas[valid.nonzero().squeeze()]
    vlabels = labels[valid]
    return vprobas, vlabels
 
def xloss(logits, labels, ignore=None):
    """
    Cross entropy loss
    """
    return F.cross_entropy(logits, Variable(labels), ignore_index=255)
 
 
# --------------------------- HELPER FUNCTIONS ---------------------------
def isnan(x):
    return x != x
    
    
def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = ifilterfalse(isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n