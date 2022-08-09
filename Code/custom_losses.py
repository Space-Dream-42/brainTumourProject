import torch.nn as nn
import torch


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

def get_loss(loss_fn, has_minicubes):
    if has_minicubes:
        if step % 4 == 0:
            # Get a new minicube batch
            batch = split_cube(train_iter.next())
    else:
        if step % 155 == 0:
            batch =