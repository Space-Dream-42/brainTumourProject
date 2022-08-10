import torch
import torch.nn as nn
from custom_losses import get_loss
from dataset_utils import split_cube, slice_cube

def train_model(model, optimizer, loss, epochs, device, has_minicubes, train_iter, steps_per_epoch = 400):
    if has_minicubes:
        save_rate = 200
    else:
        save_rate = 1600
    # training settings
    losses = []
    model.train()
    # training loop
    for epoch in range(epochs):
        for step in range(steps_per_epoch):
            if has_minicubes: #check if model is 3D
                if step % 4 == 0:
                    # Get a new minicube batch
                    batch = split_cube(train_iter.next())
            else:
                if step % 160 == 0: #check if model is 2D
                    batch = slice_cube(train_iter.next())
            loss = get_loss(model, loss, has_minicubes, step, device, batch)
            losses.append(loss)

            if step % save_rate == 0:
                print(f'epoch {epoch}: step {step:3d}: loss={loss:3.3f}')
                path = f'../Weights/weights_epoch{epoch}_step{step}_loss{loss:3.3f}.h5'
                torch.save(model.state_dict(), path)

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return losses
