import torch
import numpy as np
import os

class BraTS_Dataset():
    """
    This custom Dataset class can be used as a training or test dataset.
    It requires that the data path is specified.
    """
    def __init__(self, path, dataset_type='training'):
        self.path = path  # this should be the root dir for extracted (or cropped) files
        self.dataset_type = dataset_type
        
        if self.dataset_type == 'training':
            self.imgTr_dir = os.path.join(path, 'imagesTr')
            self.labelsTr_dir = os.path.join(path, 'labelsTr')
        elif self.dataset_type == 'test':
            self.imgTr_dir = os.path.join(path, 'imagesTs')
            self.labelsTr_dir = os.path.join(path, 'labelsTs')

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
        Loads and returns a single sample at a given index (idx) as a dictionary 
        with keys 'image' and 'label' that includes the according numpy arrays.
        """
        if self.dataset_type == 'training':
            idx = idx % 400
        
        if self.dataset_type == 'test':
            idx = idx % 84
            
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
            'label': label,
            'idx': idx
        }
        return item

    def __len__(self):
        """
        Returns the number of training-files and checks
        if it is equal to the number of training-labels.
        """
        assert len(self.imagesTr) == len(self.labelsTr)
        return len(self.imagesTr)


def get_train_test_iters(path, batch_size=1, shuffle=True, num_workers=0):
    """
    Returns data iterators for the custom dataset.
    The iterators need to be reset (call this function again) after each epoch of training.
    """
    train_data = BraTS_Dataset(path, dataset_type='training')
    test_data = BraTS_Dataset(path, dataset_type='test')
    
    trainloader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    train_iter = iter(trainloader)
    test_iter = iter(testloader)
    return train_iter, test_iter