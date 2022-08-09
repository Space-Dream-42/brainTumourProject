import torch


def train_model(model, optimizer, loss, epochs,  steps_per_epoch = 400):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # training settings
    losses = []

    # training loop
    for epoch in range(epochs):
        for step in range(steps_per_epoch):

            if step % 4 == 0:
                # Get a new minicube batch
                minicube_batch = split_cube(train_iter.next())

            loss = get_minicube_batch_loss(minicube_batch, step, device=device)
            losses.append(loss)

            if step % 20 == 0:
                print(f'epoch {epoch}: step {step:3d}: loss={loss:3.3f}')
                path = f'../Weights/weights_epoch{epoch}_step{step}_loss{loss:3.3f}.h5'
                torch.save(model.state_dict(), path)

            # backprop loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()