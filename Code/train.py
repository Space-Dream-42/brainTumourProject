import torch
import torch.nn as nn
import numpy as np
from custom_losses import get_loss
from dataset_utils import split_cube, slice_cube, split_cube_with_context

def train_model(model, optimizer, loss_fn, epochs, device, train_3d, train_iter, test_iter, compute_test_loss, batches_per_epoch=400):
    """
    Trains the given 2D or 3D model and saves the trained weights once per epoch.
    Returns the train and test losses.
    """
    num_test_samples = 2 # 1 is only for testing, should be 84

    if train_3d:
        steps_per_epoch = batches_per_epoch * 4     # multiply by minicube batches per batch
        test_steps = num_test_samples * 4              
    else:
        steps_per_epoch = batches_per_epoch * 160   # multiply by slices per batch
        test_steps = num_test_samples * 160

    # training settings
    model.train()
    train_losses = []
    test_losses = [] if compute_test_loss else None

    # training loop
    for epoch in range(epochs):
        epoch_train_losses = []

        for step in range(steps_per_epoch):  
            # 3D model
            if train_3d:
                if step % 4 == 0:
                    batch = split_cube_with_context(train_iter.next()) # Get a new batch of 3D minicubes
            
            # 2D model
            else:
                if step % 160 == 0:
                    batch = slice_cube(train_iter.next()) # Get a new batch of 2D slices

            loss = get_loss(model, loss_fn, train_3d, step, device, batch)
            epoch_train_losses.append(loss.detach().cpu().numpy())

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # calculate mean training loss of epoch
        mean_train_loss = np.mean(epoch_train_losses)
        train_losses.append(mean_train_loss) 

        # calculate mean test loss of epoch
        if compute_test_loss:
            epoch_test_losses = []

            for step in range(test_steps):
                # 3D model
                if train_3d:
                    if step % 4 == 0:
                        batch = split_cube(test_iter.next()) # Get a new batch of 3D minicubes
            
                # 2D model
                else:
                    if step % 160 == 0:
                        batch = slice_cube(test_iter.next()) # Get a new batch of 2D slices
                
                loss = get_loss(model, loss_fn, train_3d, step, device, batch)
                epoch_test_losses.append(loss.detach().cpu().numpy())

            mean_test_loss = np.mean(epoch_test_losses)
            test_losses.append(mean_test_loss) 
        
        # print train and test loss of epoch
        if compute_test_loss:
            print(f'epoch {epoch}: epoch_train_loss={mean_train_loss:3.3f}, epoch_test_loss={mean_test_loss:3.3f}')
        else:
            print(f'epoch {epoch}: epoch_train_loss={mean_train_loss:3.3f}')

        # save the model
        path = f'../Weights/{model.__class__.__name__}_epoch{epoch}_step{step}_loss{mean_train_loss:3.3f}.h5'
        torch.save(model.state_dict(), path)

    return train_losses, test_losses
