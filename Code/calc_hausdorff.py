import os
import torch
from Architectures.unet_3d import UNet3D
from Architectures.unet_3d_context import UNet3D_Mini
from Architectures.unet_2d import UNet2D
from data_loading import get_train_test_iters
from custom_losses import hausdorff_loss
from dataset_utils import segment_entire_3d_cube, predict_whole_cube_2d
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))

def load_model(architecture, target_model):
    weights_path = os.path.join('..','Weights')
    weights_filenames = {'UNet3D':['UNet3D_epoch19_loss3.674_defaultVals.h5','UNet3D_epoch19_loss3.752.h5', 'UNet3D_epoch99_loss3.752.h5'], 
                        'UNet3D_Mini': ['UNet3D_Mini_epoch38_loss3.674.h5'], 'UNet2D': []}

    models = {'UNet3D':{}, 'UNet3D_Mini': {}, 'UNet2D': {}}
    for model in weights_filenames:
        for file_name in weights_filenames[model]:
            if model == 'UNet3D_Mini':
                models[model][file_name] = UNet3D_Mini(num_modalities=4, num_classes=4).to(device)
            elif model == 'UNet3D':
                models[model][file_name] = UNet3D(num_modalities=4, num_classes=4).to(device)
            elif model == 'UNet2D':
                models[model][file_name] = UNet2D().to(device)
            else:
                print("You are requesting a model that is not available")

    inference_model = models[architecture][target_model]
    inference_model.load_state_dict(torch.load(os.path.join(weights_path, target_model), map_location=torch.device('cpu')))
    inference_model.eval()

    return inference_model


def calc_loss(model, data_iter, architecture= "UNet3D"):
    loss = 0
    dataset_size = 0

    for instance in data_iter:
        if architecture == "UNet3D":
            pred = segment_entire_3d_cube(model, instance, False, device)[2:157]
        elif architecture == "UNet3D_Mini":
            pred = segment_entire_3d_cube(model, instance, True, device)[2:157]
        elif architecture == "UNet2D":
            pred = predict_whole_cube_2d(model, instance, device)[2:157]
        pred = pred.numpy()
        label = instance['label'].numpy()[0,0]
        loss_list = hausdorff_loss(inputs = pred, targets=label)
        loss += sum(loss_list)/len(loss_list)
        dataset_size += 1

    loss /= dataset_size

    return loss

def main():
    models = {'UNet3D':['UNet3D_epoch19_loss3.674_defaultVals.h5','UNet3D_epoch19_loss3.752.h5', 'UNet3D_epoch99_loss3.752.h5'], 
                        'UNet3D_Mini': ['UNet3D_Mini_epoch38_loss3.674.h5'], 'UNet2D': []}

    # calc the losses
    batch_size = 1
    dataset_path = os.path.join('..', 'Task01_BrainTumour', 'cropped')
    losses_train = {'UNet3D': {}, 'UNet3D_Mini': {}, 'UNet2D': {}}
    losses_test = {'UNet3D': {}, 'UNet3D_Mini': {}, 'UNet2D': {}}

    for architecture in models:
        print("Architecture: " + architecture)
        for file_name in models[architecture]:
            print("Target model: " + file_name)
            model = load_model(architecture, file_name)
            train_iter, test_iter = get_train_test_iters(dataset_path, batch_size=batch_size, shuffle=False, num_workers=0)

            train_loss = calc_loss(model, train_iter, architecture=architecture)
            test_loss = calc_loss(model, test_iter, architecture=architecture)
            print("train_loss: " + train_loss)
            print("test_loss: " + test_loss)

            losses_train[architecture][file_name] = train_loss
            losses_test[architecture][file_name] = test_loss

            del model


    # store the loss in a file
    with open('losses_train.json', 'w') as fp:
        json.dump(losses_train, fp)

    with open('losses_test.json', 'w') as fp:
        json.dump(losses_test, fp)


if __name__ == '__main__':
    main()
