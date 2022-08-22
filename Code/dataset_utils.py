# This file provides the following utility functions and classes:
# split_cube function for getting minicubes
# plot_batch function
# crop_batch and decrop_batch functions
# ...

import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib.colors import from_levels_and_colors
from matplotlib.animation import FuncAnimation
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import numpy as np


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

def split_cube(input_batch, add_context):
    """
    Takes a batch of 3d cubes (with different modalities) as input and
    splits each cube into 8 patches (each cube dimension is halved).
    If add_context is set to True, overlaps of 20 pixels between patches are 
    added in each direction, so that context is preserved at the splitting borders.
    If the batchsize is > 1 then each minicube_batch contains the minicubes
    from that position of each big cube in the input batch.
    Returns a batch of 3d Minicubes.
    """
    minicube_batch = dict()

    # copy tensor so that the function has no side effects
    batch = dict()
    batch['image'] = input_batch['image'].clone()
    batch['label'] = input_batch['label'].clone()
    
    # add zero-padding slices
    image_padding = (0,0, 0,0, 3,2, 0,0, 0,0)
    label_padding = (0,0, 0,0, 3,2, 0,0)
    batch['image'] = F.pad(batch['image'], image_padding, 'constant', 0)
    batch['label'] = F.pad(batch['label'], label_padding, 'constant', 0)
    
    if add_context:
        # split images with context (20px added for each axis)
        minicube_batch0 = batch['image'][:, :, :100, :116, :116]
        minicube_batch1 = batch['image'][:, :, :100, :116,  76:]
        minicube_batch2 = batch['image'][:, :, :100,  76:, :116]
        minicube_batch3 = batch['image'][:, :, :100,  76:,  76:]
        
        minicube_batch4 = batch['image'][:, :, 60:, :116, :116]
        minicube_batch5 = batch['image'][:, :, 60:, :116,  76:]
        minicube_batch6 = batch['image'][:, :, 60:,  76:, :116]
        minicube_batch7 = batch['image'][:, :, 60:,  76:,  76:]

    else:   
        # split images without context
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


def concat_minicubes(segmented_minicubes):
    
    # concatenate along width-axis 
    lower_part_0 = torch.cat((segmented_minicubes[0], segmented_minicubes[1]), dim=2)
    lower_part_1 = torch.cat((segmented_minicubes[2], segmented_minicubes[3]), dim=2)
    upper_part_0 = torch.cat((segmented_minicubes[4], segmented_minicubes[5]), dim=2)
    upper_part_1 = torch.cat((segmented_minicubes[6], segmented_minicubes[7]), dim=2)
    
    # concatenate along 
    whole_lower_part = torch.cat((lower_part_0, lower_part_1), dim=1)
    whole_upper_part = torch.cat((upper_part_0, upper_part_1), dim=1)
    
    # concatenate along height-axis
    segmented_cube = torch.cat((whole_lower_part, whole_upper_part), dim=0)
    return segmented_cube


def segment_entire_3d_cube(model, batch, add_context, device):
    minicube_batch = split_cube(batch, add_context) # split cubes into minicubes
    sm = nn.Softmax(dim=1)
    
    for minicube_idx in range(8):
        x = minicube_batch['image'][None,minicube_idx,:,:,:,:].to(device)
        voxel_logits = model.forward(x).cpu()
        voxel_probs = sm(voxel_logits)
        probs, out = torch.max(voxel_probs, dim=1)
        if minicube_idx == 0:
            segmented_minicubes = out
        else:
            segmented_minicubes = torch.cat((segmented_minicubes, out), dim=0)
    
    segmented_cube = concat_minicubes(segmented_minicubes)
    return segmented_cube


def plot_batch(batch, num_rows=2, height=70):
    
    # plt.clf()
    fig, ax_array = plt.subplots(num_rows, 4, figsize=(12,6), 
                                 gridspec_kw = {'wspace':0, 'hspace':0})
    
    for i, ax in enumerate(fig.axes):
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for row in range(num_rows):
            x = batch['image'][row]
            
            # modalities
            indices = [col_idx + (4*row) for col_idx in range(4)]
            if i in indices:
                ax.imshow(x[i%4, height, :, :], cmap="gray", origin="lower")
                
    plt.show()
    #plt.close()


def _plot_slice(pred_slice, label_slice, height=70):
    colors = [(0.3,0.4,0.7),(0.1, 0.9, 0.5),(0.9,0.7,0.2), (0.9,0.4,0.0)]
    
    plt.rcParams.update({'axes.labelsize': 14})
    
    cmap, norm = from_levels_and_colors([0,1,2,3,4], colors)
    slice_labels = ['prediction', 'label']
    fig, axes = plt.subplots(ncols=2)
    fig.set_size_inches(10, 5)

    
    for ax, data, slice_label in zip(axes, [pred_slice, label_slice], slice_labels):
        im = ax.imshow(data, 
                       cmap = cmap,
                       norm = norm, 
                       interpolation ='none')
        ax.set(xlabel=slice_label)
        ax.tick_params(labelsize=12)
    
    cbar = fig.colorbar(im, ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels([Labels[i] for i in range(4)], fontsize=12)
    plt.show()
    plt.close()


def predict_whole_cube_2d(model, batch, device):
    image_to_predict_on = slice_cube(batch)['image']
    prediction = torch.zeros((160, 192, 192))
    for i in range(160):
        probs, out = torch.max(torch.sigmoid(model.forward(image_to_predict_on[i].to(device)).cpu()), dim=1)
        prediction[i] = out[0]
    return prediction


def animate_cube(model, batch, add_context, device, is_3d=True):
    if is_3d:
        pred_cube = segment_entire_3d_cube(model, batch, add_context, device).cpu()
    else:
        pred_cube = predict_whole_cube_2d(model, batch, device)
    label_slice = batch['label'][0, 0, :, :, :].cpu()
    image_height = len(pred_cube)
    colors = [(0.3, 0.4, 0.7), (0.1, 0.9, 0.5), (0.9, 0.7, 0.2), (0.9, 0.4, 0.0)]
    cmap, norm = from_levels_and_colors([0, 1, 2, 3, 4], colors)

    fig = plt.figure(tight_layout=True)
    gs = GridSpec(2, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax4 = fig.add_subplot(gs[0, 3])
    ax5 = fig.add_subplot(gs[1, :2])
    ax6 = fig.add_subplot(gs[1, 2:])
    axes = fig.axes
    fig.set_size_inches(10, 10)
    plt.rcParams.update({'axes.labelsize': 14})

    def animation_step_slice(height):
        height = height*4
        current_pred_slice = pred_cube[height]
        current_label_slice = label_slice[height]

        for i, ax in enumerate(axes[:4]):
            ax.clear()
            ax.imshow(batch['image'][0, i, height, :, :], cmap="gray")

        slice_labels = ['prediction', 'label']
        for ax, data, slice_label in zip(axes[4:], [current_pred_slice, current_label_slice], slice_labels):
            ax.clear()
            ax.imshow(data, cmap=cmap, norm=norm, interpolation='none')
            ax.set(xlabel=slice_label)
            ax.tick_params(labelsize=12)

    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ticks=[0, 1, 2, 3], orientation='vertical')
    cbar.ax.set_yticklabels([Labels[i] for i in range(4)], fontsize=12)

    ani = FuncAnimation(fig, animation_step_slice, frames=int(image_height/4)-2, interval=100, repeat=True)
    return ani


def plot_minicube_pred_label(model, minicube_batch, device, minicube_idx, height=70):
    """
    Takes a batch of minicubes as input and outputs a comparison plot between a slice 
    of the prediction on a minicube and the label of the minicube
    at the given height of the minicube at the given index in the batch.
    """
    
    # make prediction for minicube
    voxel_logits_batch = model.forward(minicube_batch['image'][None, minicube_idx,:,:,:,:].to(device))
    
    sm = nn.Softmax(dim=1)
    voxel_probs_batch = sm(voxel_logits_batch)
    probs, out = torch.max(voxel_probs_batch, dim=1)
    
    pred_slice = out[0, height, :, :].cpu()
    label_slice = minicube_batch['label'][minicube_idx, 0, height, :, :].cpu()
    
    _plot_slice(pred_slice, label_slice, height)


def plot_cube_pred_label(model, batch, add_context, device, height=70):
    """
    Takes a raw 3d batch as input and outputs a comparison plot between 
    a slice of the prediction and the label at the given height.
    """
    # make prediction on entire cube
    segmented_cube = segment_entire_3d_cube(model, batch, add_context, device)
    
    pred_slice = segmented_cube[height, :, :].cpu()
    label_slice = batch['label'][0, 0, height, :, :].cpu()
    
    _plot_slice(pred_slice, label_slice, height)


def plot_loss(train_losses, test_losses):
    plt.plot(train_losses, label='training loss')
    if test_losses is not None:
        plt.plot(test_losses, label='test loss')
    plt.xlabel('epoch', fontsize=14)
    plt.ylabel('loss', fontsize=14)
    plt.legend()
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


def center_crop(z,x,y,img):
    to_crop_z = img.shape[2] - z
    to_crop_x = img.shape[3] - x
    to_crop_y = img.shape[4] - y
    get_bound = lambda x,length: (x//2,(length-x//2)) if x%2 == 0 else (x//2 + 1, length-x//2)
    # switch to torch.div
    # get_bound = lambda x,length: (torch.div(x, 2, 'floor'),length-torch.div(x//2)) if x%2 == 0 else (torch.div(x, 2, 'floor') + 1, length-torch.div(x//2))
    slice_z = get_bound(to_crop_z,img.shape[2])
    slice_x = get_bound(to_crop_x,img.shape[3])
    slice_y = get_bound(to_crop_y,img.shape[4])
    return img[:,:,slice_z[0]:slice_z[1],slice_x[0]:slice_x[1],slice_y[0]:slice_y[1]]


def plot_confusion_matrix(test_iter, model, train_3d, add_context, device):
    y_pred = []
    y_true = []
    for i in range(3):
        batch = test_iter.next()
        if train_3d:
            batch = split_cube(batch, add_context)
        else:
            batch = slice_cube(batch)
        inputs = batch['image'].to(device)
        labels = batch['label'].to(device)
        output = model(inputs)
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output)
        labels = labels.data.cpu().numpy()
        y_true.extend(labels)
    classes = [Labels[i] for i in range(4)]
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix) * 10, index=[i for i in classes], columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    return plt

