# This file provides the following utility functions and classes:
# split_cube function for getting minicubes
# plot_batch function
# crop_batch and decrop_batch functions
# ...

import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt


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
    del input_batch #
    
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
    del batch #
    
    # assemble labels into batch
    minicube_batch['label'] = torch.cat(
        (label_batch0, label_batch1, label_batch2, label_batch3, 
         label_batch4, label_batch5, label_batch6, label_batch7), 0)
    
    return minicube_batch


def slice_cube(batch):
    """
    Pads the image and puts the height dimension first to make iterating through one image easier.
    """
    twod_images_batch = dict()

    # add zero-padding slices
    image_padding = (0,0, 0,0, 3,2, 0,0, 0,0)
    label_padding = (0,0, 0,0, 3,2, 0,0)
    batch['image'] = F.pad(batch['image'], image_padding, 'constant', 0)
    batch['label'] = F.pad(batch['label'], label_padding, 'constant', 0)
    twod_images_batch['image'] = batch['image'].permute(2, 0, 1, 3, 4)
    twod_images_batch['label'] = batch['label'].permute(2, 0, 1, 3, 4)
    del batch #
    return twod_images_batch


def concat_minicubes(segmented_minicubes):
    
    # concatenate along dim 2
    lower_part_0 = torch.cat((segmented_minicubes[0], segmented_minicubes[1]), dim=2)
    lower_part_1 = torch.cat((segmented_minicubes[2], segmented_minicubes[3]), dim=2)
    upper_part_0 = torch.cat((segmented_minicubes[4], segmented_minicubes[5]), dim=2)
    upper_part_1 = torch.cat((segmented_minicubes[6], segmented_minicubes[7]), dim=2)
    
    # concatenate along dim 1
    whole_lower_part = torch.cat((lower_part_0, lower_part_1), dim=1)
    whole_upper_part = torch.cat((upper_part_0, upper_part_1), dim=1)
    
    # concatenate along dim 0
    segmented_cube = torch.cat((whole_lower_part, whole_upper_part), dim=0)
    return segmented_cube


def segment_entire_3d_cube(model, batch, add_context, device):
    """
    Applies 3d image segementation to one MRI image.
    """
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


def predict_whole_cube_2d(model, batch, device):
    """
    Segments one MRI image into slices, do 2D segmentation on them and concat them back to one image-batch.
    """
    image_to_predict_on = slice_cube(batch)['image']
    prediction = torch.zeros((160, 192, 192))
    for i in range(160):
        probs, out = torch.max(torch.sigmoid(model.forward(image_to_predict_on[i].to(device)).cpu()), dim=1)
        prediction[i] = out[0]
    return prediction

    
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
    """
    Crops the image so the output image has the shape (x,y,z)
    and is as much in the center of the original image as possible.
    """
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


def get_minicube_prediction(model, minicube_batch, step, device):
    """
    Takes two minicubes from one image each step and return their predictions.
    """
    #number_of_cubes = int(minicube_batch['image'].shape[0]/4)
    voxel_logits_batch = model.forward(minicube_batch['image'][step, :, :, :, :].to(device))

    return voxel_logits_batch
