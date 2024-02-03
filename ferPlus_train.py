import os
import sys
import argparse
import itertools
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from torchvision import transforms, datasets

import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

eps = sys.float_info.epsilon

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fer_path', type=str, default='./data/train', help='Path to FER+ dataset')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate for SGD')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=80, help='Total training epochs')
    parser.add_argument('--num_head', type=int, default=2, help='Number of attention heads')

    return parser.parse_args()


class AttentionLoss(nn.Module):
    def __init__(self, ):
        super(AttentionLoss, self).__init__()

    def forward(self, x):
        num_head = len(x)

        loss = 0
        count = 0
        if num_head > 1:
            for i in range(num_head - 1):
                for j in range(i+1, num_head):
                    mse = F.mse_loss(x[i], x[j])
                    count = count + 1
                    loss = loss + mse
            loss = count / (loss + eps)
        else:
            loss = 0
        return loss
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=16)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]*100, fmt) + '%',
                 horizontalalignment='center',
                 color='white' if cm[i, j] > thresh else 'black')
        
    plt.ylabel('Actual', fontsize=18)
    plt.xlabel('Predicted', fontsize=18)
    plt.tight_layout()

classes = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Angry', 'Contempt']