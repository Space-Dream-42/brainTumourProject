import os
import torch
import torch.nn as nn
import numpy as np
from custom_losses import get_loss
from dataset_utils import split_cube, slice_cube
from data_loading import get_train_test_iters

def train_model(model, optimizer, loss_fn, epochs, device, dataset_path, batch_size, train_3d, add_context, compute_test_loss):
    """
    Trains the given 2D or 3D model and saves the trained weights once per epoch.
    Returns the train and test losses.
    """
    # training settings
    model.train()
    train_losses = []
    test_losses = [] if compute_test_loss else None

    # training loop
    for epoch in range(epochs):
        
        train_iter, test_iter = get_train_test_iters(dataset_path, batch_size=batch_size, shuffle=False, num_workers=0)
        epoch_train_losses = []

        for step, raw_batch in enumerate(train_iter):
            #print(step)
            # 3D model
            if train_3d:
                batch = split_cube(raw_batch, add_context)  # Get a new batch of 3D minicubes
                for step_within_image in range(4):
                    loss = get_loss(model, loss_fn, train_3d, step_within_image, device, batch)
                    epoch_train_losses.append(loss.detach().cpu().numpy())
                    
                    # backprop loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            # 2D model
            else:
                batch = slice_cube(raw_batch)  # Get a new batch of 2D slices
                for step_within_image in range(160):
                    loss = get_loss(model, loss_fn, train_3d, step_within_image, device, batch)
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

            for step, raw_batch in enumerate(test_iter):
                # 3D model
                if train_3d:
                    batch = split_cube(raw_batch, add_context)  # Get a new batch of 3D minicubes
                    for step_within_image in range(4):
                        loss = get_loss(model, loss_fn, train_3d, step_within_image, device, batch)
                        epoch_test_losses.append(loss.detach().cpu().numpy())
                # 2D model
                else:
                    batch = slice_cube(raw_batch)  # Get a new batch of 2D slices
                    for step_within_image in range(160):
                        loss = get_loss(model, loss_fn, train_3d, step_within_image, device, batch)
                        epoch_test_losses.append(loss.detach().cpu().numpy())

            mean_test_loss = np.mean(epoch_test_losses)
            test_losses.append(mean_test_loss) 
        
        # print train and test loss of epoch
        if compute_test_loss:
            print(f'epoch {epoch}: epoch_train_loss={mean_train_loss:3.3f}, epoch_test_loss={mean_test_loss:3.3f}')
        else:
            print(f'epoch {epoch}: epoch_train_loss={mean_train_loss:3.3f}')

        # save the model
        weights_path = os.path.join('..', 'Weights')
        if not os.path.exists(weights_path):
            os.makedirs(weights_path)
        path = os.path.join('..', 'Weights', f'{model.__class__.__name__}_epoch{epoch}_loss{mean_train_loss:3.3f}.h5')
        torch.save(model.state_dict(), path)
    np.save(os.path.join('..', 'Losses', f'{model.__class__.__name__}_train_loss'), train_losses)
    np.save(os.path.join('..', 'Losses', f'{model.__class__.__name__}_test_loss'), test_losses)
    return train_losses, test_losses
