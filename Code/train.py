import torch
import torch.nn as nn
import numpy as np
from custom_losses import get_loss
from dataset_utils import split_cube, slice_cube

def train_model(model, optimizer, loss_fn, epochs, device, train_3d, train_iter, test_iter, compute_test_loss, batches_per_epoch=400):
    num_test_samples = 1 # 1 is only for testing, should be 84

    if train_3d:
        save_rate = 200
        steps_per_epoch = batches_per_epoch * 4     # multiply by minicube batches per batch
        test_steps = num_test_samples * 4              
    else:
        save_rate = 1600
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
                    batch = split_cube(train_iter.next()) # Get a new batch of 3D minicubes
            
            # 2D model
            else:
                if step % 160 == 0:
                    batch = slice_cube(train_iter.next()) # Get a new batch of 2D slices

            loss = get_loss(model, loss_fn, train_3d, step, device, batch)
            epoch_train_losses.append(loss.detach().cpu().numpy())

            if step % save_rate == 0:
                print(f'epoch {epoch}: step {step:3d}: loss={loss:3.3f}')
                path = f'../Weights/{model.__class__.__name__}_epoch{epoch}_step{step}_loss{loss:3.3f}.h5'
                torch.save(model.state_dict(), path)

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # calculate mean training loss of epoch
        train_losses.append(np.mean(epoch_train_losses)) 

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

            test_losses.append(np.mean(epoch_test_losses)) 

    return train_losses, test_losses
