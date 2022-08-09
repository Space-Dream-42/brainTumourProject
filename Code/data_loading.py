import torch
import numpy as np
import os

class BraTS_Dataset():
    def __init__(self, path):
        self.path = path  # this should be the root dir for extracted (or cropped) files
        self.imgTr_dir = os.path.join(path, 'imagesTr')
        self.labelsTr_dir = os.path.join(path, 'labelsTr')  # TODO: rename TR-related names

        print(self.imgTr_dir)
        print(self.labelsTr_dir)

        self.imagesTr = []
        self.labelsTr = []

        self.imagesTr_paths = []
        self.labelsTr_paths = []

        self.image_names = os.listdir(self.imgTr_dir)

        for img_name in self.image_names:
            self.imagesTr.append(img_name)
            self.labelsTr.append(img_name)

    def __getitem__(self, idx):
        """
        Loads and returns a single sample in the form of a dictionary
        with keys 'image' and 'label'.
        The function returns the sample at the given index (idx)
        by finding the path and returning the numpy array.
        """
        # load image
        img_filename = self.imagesTr[idx]
        img_path = os.path.join(self.imgTr_dir, img_filename)
        img = np.load(img_path)

        # load label
        label_filename = self.labelsTr[idx]
        label_path = os.path.join(self.labelsTr_dir, label_filename)
        label = np.load(label_path)
        label = np.expand_dims(label, axis=0)

        item = {
            'image': img,
            'label': label
        }
        return item

    def __len__(self):
        """
        Returns the number of training-files and checks
        if it is equal to the number of training-labels.
        """
        assert len(self.imagesTr) == len(self.labelsTr)
        return len(self.imagesTr)


def get_train_test_iters(training_path, test_path, batch_size=1, shuffle=True, num_workers=0):
    train_data = BraTS_Dataset(training_path)
    test_data = BraTS_Dataset(test_path)
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    train_iter = iter(trainloader)
    test_iter = iter(testloader)
    return train_iter, test_iter
