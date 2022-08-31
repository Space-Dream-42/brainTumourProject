import torch.nn as nn
import torch
from dataset_utils import split_cube, slice_cube
import seg_metrics.seg_metrics as sg

## Gratefully borrowed (with some modification) from 
## https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch

class DiceLoss(nn.Module):
    """
    Computes the dice-loss for each output-channel of the model.
    The label has to be split into 4 images to achieve this.
    """
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets):
        loss_count = torch.zeros(4)
        sm = nn.Softmax(dim=1)
        inputs = sm(inputs.float())
        for i in range(len(inputs[0])):
            current_class = targets.clone()
            current_class[current_class != i] = 10
            current_class[current_class == i] = 11
            current_class = current_class-10
            loss_count[i] = dice_loss_one_image(inputs[:, i], current_class)*(i*2+1)
        return loss_count.sum() / 16


def dice_loss_one_image(inputs, targets, smooth=1):

    # flatten label and prediction tensors
    inputs = inputs.contiguous().view(-1)
    targets = targets.contiguous().view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return (1 - dice)


class FocalTverskyLoss(nn.Module):
    """
    Computes the focal-tversky-loss for each output-channel of the model.
    The label has to be split into 4 images to achieve this.
    """
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets):
        loss_count = 0
        for i in range(len(inputs[0])):
            current_class = targets.clone()
            current_class[current_class != i] = 10
            current_class[current_class == i] = 11
            current_class = current_class - 10
            loss_count += focaltversky_loss_one_image(inputs[:, i], current_class)*(i*2+1)
        return loss_count / 16


def focaltversky_loss_one_image(inputs, targets, smooth=1, alpha=0.7, beta=0.3, gamma=4/3):
    """
    Returns the focaltversky loss for a given 2d slice.
    """
    inputs = torch.sigmoid(inputs)

    # flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    # True Positives, False Positives, False Negatives
    TP = (inputs * targets).sum()    
    FP = ((1-targets) * inputs).sum()
    FN = (targets * (1-inputs)).sum()
    
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
    FocalTversky = (1 - Tversky)**gamma
                    
    return FocalTversky


def get_minicube_batch_loss(model, loss_fn, minicube_batch, step, device):
    """
    Returns the loss for two minicubes from a 3d MR scan.
    The position of the minicubes is determined by the current step.
    """
    number_of_cubes = int(minicube_batch['image'].shape[0]/4)

    # CrossEntropyLoss (needs different dimensions for the label than all other losses)
    if str(loss_fn) == 'CrossEntropyLoss()':
        if step % 4 == 0:
            voxel_logits_batch = model.forward(minicube_batch['image'][:number_of_cubes, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][:number_of_cubes, 0, :, :, :].long().to(device))

        elif step % 4 == 1:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes:number_of_cubes*2, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes:number_of_cubes*2, 0, :, :, :].long().to(device))

        elif step % 4 == 2:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*2:number_of_cubes*3, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*2:number_of_cubes*3, 0, :, :, :].long().to(device))

        else:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*3:, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*3:, 0, :, :, :].long().to(device))
    
    # Other loss functions
    else:
        if step % 4 == 0:
            voxel_logits_batch = model.forward(minicube_batch['image'][:number_of_cubes, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][:number_of_cubes, :, :, :].long().to(device))

        elif step % 4 == 1:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes:number_of_cubes*2, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes:number_of_cubes*2, :, :, :].long().to(device))

        elif step % 4 == 2:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*2:number_of_cubes*3, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*2:number_of_cubes*3, :, :, :].long().to(device))

        else:
            voxel_logits_batch = model.forward(minicube_batch['image'][number_of_cubes*3:, :, :, :, :].to(device))
            loss = loss_fn(voxel_logits_batch, minicube_batch['label'][number_of_cubes*3:, :, :, :].long().to(device))
    del voxel_logits_batch
    return loss


def hausdorff_loss(inputs, targets):
    """
    Returns the hausdorff loss.
    """
    return sg.write_metrics(labels=[0,1,2,3],gdth_img=targets,pred_img=inputs,metrics='hd95')[0]['hd95']


def get_loss(model, loss_fn, has_minicubes, step, device, batch):
    """
    This high-level function adapts the calculation of the loss according to the used model.
    Returns the loss for a batch according to the specified loss function.
    """
    if has_minicubes:
        loss = get_minicube_batch_loss(model, loss_fn, batch, step, device)
    else:
        prediction_batch = model.forward(batch['image'][step%160, :, :, :, :].to(device))
        loss = loss_fn(prediction_batch, batch['label'][step%160, :, :, :, :].to(device))
    return loss
