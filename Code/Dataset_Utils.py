# This file provides the following utility functions and classes:
# BraTS_TrainingDataset class
# split_cube function for getting minicubes
# plot_batch function
# crop_batch and decrop_batch functions

import torch
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.datasets.vision import StandardTransform
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import pandas as pd 
from PIL import Image
import matplotlib.pyplot as plt 
import json
import os

import SimpleITK as sitk


class BraTS_TrainingDataset():
    def __init__(self, path):
        self.path = path # this should be the root dir for extracted (or cropped) files
        self.imgTr_dir = os.path.join(path, 'imagesTr')
        self.labelsTr_dir = os.path.join(path, 'labelsTr')
        
        print(self.imgTr_dir)
        print(self.labelsTr_dir)
        
        self.imagesTr = []
        self.labelsTr = []
        
        self.imagesTr_paths = []
        self.labelsTr_paths = []
        
        for img, label in zip(os.listdir(self.imgTr_dir), os.listdir(self.labelsTr_dir)):
            self.imagesTr.append(img)
            self.labelsTr.append(label)
        
    def __getitem__(self, idx):
        """
        Loads and returns a single sample in the form of a dictionary
        with keys 'image' and 'label'.
        The function returns the sample at the given index (idx)
        by finding the path and returning the numpy array.
        """
        # load image
        img_filename = self.imagesTr[idx]
        img_path = os.path.join(self.imgTr_dir, img_filename)
        img = np.load(img_path)
        
        # load label
        label_filename = self.labelsTr[idx]
        label_path = os.path.join(self.labelsTr_dir, label_filename)
        label = np.load(label_path)
        
        item = {
            'image': img,
            'label': label
        }
        return item
    
    def __len__(self):
        """
        Returns the number of training-files and checks
        if it is equal to the number of training-labels.
        """
        assert len(self.imagesTr) == len(self.labelsTr)
        return len(self.imagesTr)


# TODO:
# BraTS_TestDataset class (e.g. 80:20 train-test split)


def split_cube(batch):
    """
    Takes a batch of 3d cubes (with different modalities) as input and
    splits each cube into 8 patches (each cube dimension is halved).
    If the batchsize is > 1 then each minicube_batch contains the minicubes
    from that position of each big cube in the input batch.
    Returns a batch of 3d Minicubes.
    """
    minicube_batch = dict()
    
    # padding:
    # add empty slice to make number of slices divisible by 2
    # -> new number of slices is 156
    image_padding = (0,0, 0,0, 1,0, 0,0, 0,0)
    label_padding = (0,0, 0,0, 1,0, 0,0)
    batch['image'] = F.pad(batch['image'], image_padding, 'constant', 0)
    batch['label'] = F.pad(batch['label'], label_padding, 'constant', 0)
    
    # split images
    minicube_batch0 = batch['image'][:, :, :78, :96,  :96]
    minicube_batch1 = batch['image'][:, :, :78, :96,   96:]
    minicube_batch2 = batch['image'][:, :, :78,  96:, :96]
    minicube_batch3 = batch['image'][:, :, :78,  96:,  96:]
    
    minicube_batch4 = batch['image'][:, :, 78:, :96,  :96]
    minicube_batch5 = batch['image'][:, :, 78:, :96,   96:]
    minicube_batch6 = batch['image'][:, :, 78:,  96:, :96]
    minicube_batch7 = batch['image'][:, :, 78:,  96:,  96:]
    
    # assemble minicubes into batch
    minicube_batch['image'] = torch.cat(
        (minicube_batch0, minicube_batch1, minicube_batch2, minicube_batch3, 
         minicube_batch4, minicube_batch5, minicube_batch6, minicube_batch7), 0)
    
    # split labels
    label_batch0 = batch['label'][:, :78, :96,  :96]
    label_batch1 = batch['label'][:, :78, :96,   96:]
    label_batch2 = batch['label'][:, :78,  96:, :96]
    label_batch3 = batch['label'][:, :78,  96:,  96:]
    
    label_batch4 = batch['label'][:, 78:, :96,  :96]
    label_batch5 = batch['label'][:, 78:, :96,   96:]
    label_batch6 = batch['label'][:, 78:,  96:, :96]
    label_batch7 = batch['label'][:, 78:,  96:,  96:]
    
    # assemble labels into batch
    minicube_batch['label'] = torch.cat(
        (label_batch0, label_batch1, label_batch2, label_batch3, 
         label_batch4, label_batch5, label_batch6, label_batch7), 0)
    
    return minicube_batch


def plot_batch(batch, num_rows=2, height=70):
    
    plt.clf()
    fig, ax_array = plt.subplots(num_rows, 5, figsize=(12,6), 
                                 gridspec_kw = {'wspace':0, 'hspace':0})
    
    for i, ax in enumerate(fig.axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for row in range(num_rows):
            x = batch['image'][row]
            y = batch['label'][row]
            
            # modalities
            indices = [col_idx + (5*row) for col_idx in range(4)]
            if i in indices:
                ax.imshow(x[i%5, height, :, :], cmap="gray", origin="lower")
                
            # label
            if i == (5*row + 4):
                ax.imshow(y[height, :, :], cmap="gray", origin="lower")
    plt.show()
    plt.close()
    
    
def crop_batch(img_batch):
    """
    Crops a batch of images to output size (192, 192) per slice.
    """
    n_dims = img_batch.dim()
    
    # Crop 2D slice
    if n_dims == 2:
        cropped_result = img_batch[19:211, 19:211]
    
    # Crop slices in 3D Volume
    elif n_dims == 3:
        cropped_result = img_batch[:, 19:211, 19:211]
        
    # Crop slices in batch of 3D Volumes
    elif n_dims == 4:
        cropped_result = img_batch[:, :, 19:211, 19:211]
        
    else:
        raise IndexError
    return cropped_result


def decrop_batch(img_batch):
    """
    Decrops a batch of images by applying zero-padding (zero is the background-label),
    but also works on just a single image.
    Output size per slice is (240, 240).
    """
    decropped_batch = F.pad(img_batch, (19,29,19,29), 'constant', 0)
    return decropped_batch
