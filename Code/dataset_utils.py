# This file provides the following utility functions and classes:
# BraTS_TrainingDataset class
# split_cube function for getting minicubes
# plot_batch function
# crop_batch and decrop_batch functions

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
import os

Scans = {
    0: 'FLAIR',
    1: 'T1w',
    2: 't1gd',
    3: 'T2w'}

Labels = {
    0: 'background',
    1: 'edema',
    2: 'non-enhancing tumor',
    3: 'enhancing tumour'}

def split_cube(batch):
    """
    Takes a batch of 3d cubes (with different modalities) as input and
    splits each cube into 8 patches (each cube dimension is halved).
    If the batchsize is > 1 then each minicube_batch contains the minicubes
    from that position of each big cube in the input batch.
    Returns a batch of 3d Minicubes.
    """
    minicube_batch = dict()
    
    # add zero-padding slices
    image_padding = (0,0, 0,0, 3,2, 0,0, 0,0)
    label_padding = (0,0, 0,0, 3,2, 0,0)
    batch['image'] = F.pad(batch['image'], image_padding, 'constant', 0)
    batch['label'] = F.pad(batch['label'], label_padding, 'constant', 0)
    
    # split images
    minicube_batch0 = batch['image'][:, :, :80, :96,  :96]
    minicube_batch1 = batch['image'][:, :, :80, :96,   96:]
    minicube_batch2 = batch['image'][:, :, :80,  96:, :96]
    minicube_batch3 = batch['image'][:, :, :80,  96:,  96:]
    
    minicube_batch4 = batch['image'][:, :, 80:, :96,  :96]
    minicube_batch5 = batch['image'][:, :, 80:, :96,   96:]
    minicube_batch6 = batch['image'][:, :, 80:,  96:, :96]
    minicube_batch7 = batch['image'][:, :, 80:,  96:,  96:]
    
    # assemble minicubes into batch
    minicube_batch['image'] = torch.cat(
        (minicube_batch0, minicube_batch1, minicube_batch2, minicube_batch3,
         minicube_batch4, minicube_batch5, minicube_batch6, minicube_batch7), 0)
    
    # split labels
    label_batch0 = batch['label'][:, :, :80, :96,  :96]
    label_batch1 = batch['label'][:, :, :80, :96,   96:]
    label_batch2 = batch['label'][:, :, :80,  96:, :96]
    label_batch3 = batch['label'][:, :, :80,  96:,  96:]
    
    label_batch4 = batch['label'][:, :, 80:, :96,  :96]
    label_batch5 = batch['label'][:, :, 80:, :96,   96:]
    label_batch6 = batch['label'][:, :, 80:,  96:, :96]
    label_batch7 = batch['label'][:, :, 80:,  96:,  96:]
    
    # assemble labels into batch
    minicube_batch['label'] = torch.cat(
        (label_batch0, label_batch1, label_batch2, label_batch3, 
         label_batch4, label_batch5, label_batch6, label_batch7), 0)
    
    return minicube_batch

def slice_cube(batch):
    twod_images_batch = dict()

    # add zero-padding slices
    image_padding = (0,0, 0,0, 3,2, 0,0, 0,0)
    label_padding = (0,0, 0,0, 3,2, 0,0)
    batch['image'] = F.pad(batch['image'], image_padding, 'constant', 0)
    batch['label'] = F.pad(batch['label'], label_padding, 'constant', 0)
    twod_images_batch['image'] = batch['image'].permute(2, 0, 1, 3, 4)
    twod_images_batch['label'] = batch['label'].permute(2, 0, 1, 3, 4)
    return twod_images_batch


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

    
def plot_pred_label_comparison(model, minicube_batch, minicube_idx, height=70):
    """
    Takes a batch of minicubes as input and outputs a comparison plot between a slice of the
    prediction and the label at the given height of the minicube at the given index in the batch.
    """
    
    # make prediction
    voxel_logits_batch = model.forward(minicube_batch['image'][None, minicube_idx,:,:,:,:])
    
    sm = nn.Softmax(dim=1)
    voxel_probs_batch = sm(voxel_logits_batch)
    probs, out = torch.max(voxel_probs_batch, dim=1)
    
    pred_slice = out[0, height, :, :].cpu()
    label_slice = minicube_batch['label'][minicube_idx, height, :, :].cpu()
    
    # color picker: https://www.tug.org/pracjourn/2007-4/walden/color.pdf
    colors = [(0.3,0.4,0.7),(0.1, 0.9, 0.5),(0.9,0.7,0.2), (0.9,0.4,0.0)]
    
    plt.rcParams.update({'axes.labelsize': 30})
    
    cmap, norm = from_levels_and_colors([0,1,2,3,4], colors)
    slice_labels = ['prediction', 'label']
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(18, 10)
    
    
    for ax, data, slice_label in zip(axes, [pred_slice, label_slice], slice_labels):
        im = ax.imshow(data, 
                       cmap = cmap,
                       norm = norm, 
                       interpolation ='none')
        ax.set(xlabel=slice_label)
        ax.tick_params(labelsize=18)
    
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels([Labels[i] for i in range(4)], fontsize=18)
    
    plt.show()
    
    
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

