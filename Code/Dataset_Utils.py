# This file provides the following utility functions and classes:
# BraTS_TrainingDataset class
# BraTS_TestDataset class
# plot_batch function

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
        self.path = path # this should be the root dir for extracted files
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


class BraTS_TestDataset():
    def __init__(self, path, device='cpu'):
        self.path = path # this should be the root dir for extracted files
        self.imgTs_dir = os.path.join(path, 'imagesTs')
        
        print(self.imgTs_dir)
        
        self.imagesTs = []
        self.imagesTs_paths = []
        
        for img in os.listdir(self.imgTs_dir):
            self.imagesTs.append(img)
            
    def __getitem__(self, idx):
        """
        Loads and returns a single sample in the form of a dictionary
        with keys 'image' and 'label'.
        The function returns the sample at the given index (idx)
        by finding the path and returning the numpy array.
        """
        # load test image
        img_filename = self.imagesTs[idx]
        img_path = os.path.join(self.imgTs_dir, img_filename)
        img = np.load(img_path)
        return img
    
    def __len__(self):
        """
        Returns the number of test-files.
        """
        return len(self.imagesTs)


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
    Crops a batch of images to output size (191, 161) per slice.
    """
    n_dims = img_batch.dim()
    
    # Crop 2D slice
    if n_dims == 2:
        cropped_result = img_batch[19:210, 38:199]
    
    # Crop slices in 3D Volume
    elif n_dims == 3:
        cropped_result = img_batch[:, 19:210, 38:199]
        
    # Crop slices in batch of 3D Volumes
    elif n_dims == 4:
        cropped_result = img_batch[:, :, 19:210, 38:199]
        
    else:
        raise IndexError
    return cropped_result


def decrop_batch(img_batch):
    """
    Decrops a batch of images by applying zero-padding (zero is the background-label),
    but also works on just a single image.
    Output size per slice is (240, 240).
    """
    decropped_batch = F.pad(out, (38,41,19,30), 'constant', 0)
    return decropped_batch
