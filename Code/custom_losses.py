import torch.nn as nn
import torch
from dataset_utils import split_cube, slice_cube

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        loss_count = 0
        for i in range(len(inputs[0])):
            current_class = targets.clone()
            current_class[current_class != i] = 0
            current_class[current_class == i] = 1
            loss_count += dice_loss_one_image(inputs[:, i], current_class)
        return loss_count / 4


def dice_loss_one_image(inputs, targets, smooth=1):
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice


def get_minicube_batch_loss(model, loss_fn, minicube_batch, step, device):
    number_of_cubes = int(minicube_batch['image'].shape[0]/4)
    #if step % 4 == 0:
    voxel_logits_batch = model.forward(minicube_batch['image'][:number_of_cubes, :, :, :, :].to(device))
    loss = loss_fn(voxel_logits_batch, minicube_batch['label'][:number_of_cubes, :, :, :].long().to(device))
    return loss

    #elif step % 4 == 1:
    #    voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes:number_of_cubes*2, :, :, :, :].to(device))
    #    loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes:number_of_cubes*2, :, :, :].long().to(device))
    #    return loss

    #elif step % 4 == 2:
    #    voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*2:number_of_cubes*3, :, :, :, :].to(device))
    #    loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*2:number_of_cubes*3, :, :, :].long().to(device))
    #    return loss

    #else:
    #    voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*3:, :, :, :, :].to(device))
    #    loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*3:, :, :, :].long().to(device))
        #return loss




def get_loss(model, loss_fn, has_minicubes, step, device, batch):
    if has_minicubes:
        loss = get_minicube_batch_loss(model, loss_fn, batch, step, device)
    else:
        prediction_batch = model.forward(batch['image'][step%160, :, :, :, :].to(device))
        loss = loss_fn(prediction_batch, batch['label'][step%160, :, :, :, :].to(device))
    return loss
