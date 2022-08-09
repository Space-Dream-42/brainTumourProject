import numpy as np
import json
import os

# TODO: PLEASE TEST!!!

import SimpleITK as sitk

# prepare extracting files:
# make new dirs etc

root_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(root_dir, os.pardir))
dataset_dir = os.path.join(parent_dir, 'Task01_BrainTumour')
print(f'root_dir:    {root_dir}')
print(f'parent_dir:  {parent_dir}')
print(f'dataset_dir: {dataset_dir}')

# imagesTr: training images
# labelsTr: training labels
# imagesTs: test images

extracted_dir = os.path.join(dataset_dir, 'extracted')
imgTr_dir = os.path.join(extracted_dir, 'imagesTr')
labelsTr_dir = os.path.join(extracted_dir, 'labelsTr')
imgTs_dir = os.path.join(extracted_dir, 'imagesTs')

if not os.path.exists(dataset_dir + '/extracted'):
    extracted = False
    print('data not extracted yet.\n -> creating new extracted dir.')
    for dir_path in [extracted_dir, imgTr_dir, labelsTr_dir, imgTs_dir]:
        os.mkdir(dir_path)
else:
    extracted = True

# load the json dataset-info-file as a dictionary
with open(dataset_dir + '/dataset.json') as json_file:
    data = json.load(json_file)
    print(data.keys())
    print('\nScans:', data['modality'])
    print('\nLabels:', data['labels'])
    print('\n#Training:', data['numTraining'])
    print('#Test:', data['numTest'])

    train_filenames = data['training']
    test_filenames = data['test']

print('\nTraining-files paths:', train_filenames[0])
print('Test-files path:', test_filenames[0])

# extract files
# needs about 110-120 GB disk-storage
# or modify the code (break after e.g. 10 iterations)
# to only unpack a couple of samples

if extracted:
    print('files are already extracted.')

else:
    num_of_train_files = len(train_filenames)
    num_of_test_files = len(test_filenames)

    for i, img_path_dict in enumerate(train_filenames):
        print(f'extracting training_file {i}/{num_of_train_files}', end="\r")
        image_path_gz = img_path_dict['image'][2:]
        label_path_gz = img_path_dict['label'][2:]

        # extract and save image
        img_path = os.path.join(dataset_dir, image_path_gz)  # nii-image path
        sitk_img = sitk.ReadImage(img_path)  # read the nii-image
        img = sitk.GetArrayFromImage(sitk_img)  # img to numpy array
        extracted_img_path = os.path.join(imgTr_dir, str(i))
        np.save(extracted_img_path, img)

        # extract and save label
        label_path = os.path.join(dataset_dir, label_path_gz)  # nii-image path
        sitk_label_img = sitk.ReadImage(label_path)  # read the nii-image
        img_label = sitk.GetArrayFromImage(sitk_label_img)  # img to numpy array
        extracted_label_path = os.path.join(labelsTr_dir, str(i))
        np.save(extracted_label_path, img_label)
    print('finished extracting training files.')

    for i, img_path in enumerate(test_filenames):
        print(f'extracting testing_file {i}/{num_of_test_files}', end="\r")
        image_path_gz = img_path[2:]

        # extract and save test image
        img_path = os.path.join(dataset_dir, image_path_gz)  # nii-image path
        sitk_img = sitk.ReadImage(img_path)  # read the nii-image
        img = sitk.GetArrayFromImage(sitk_img)  # img to numpy array
        extracted_img_path = os.path.join(imgTs_dir, str(i))
        np.save(extracted_img_path, img)
    print('finished extracting test files.')

extracted_directory = '../Task01_BrainTumour/extracted/'
cropped_directory = '../Task01_BrainTumour/cropped/'
image_directory = 'imagesTr/%i.npy'
label_directory = 'labelsTr/%i.npy'


def get_image_and_target(i):
    return np.load(imgTr_dir + '%i.npy' % i), np.load(labelsTr_dir + '%i.npy' % i)


if not os.path.exists('../Task01_BrainTumour/cropped/'):
    print('data not extracted yet.\n -> creating new extracted dir.')
    os.mkdir('../Task01_BrainTumour/cropped/')
    os.mkdir('../Task01_BrainTumour/cropped/imagesTr/')
    os.mkdir('../Task01_BrainTumour/cropped/labelsTr/')


for i in range(484):
    current_image, current_label = get_image_and_target(i)
    np.save(cropped_directory + image_directory % i, current_image[:, :, 19:211, 19:211])
    np.save(cropped_directory + label_directory % i, current_label[:, 19:211, 19:211])
    print(f'cropping images: {i}/483', end="\r")

# TODO: Test-Train-Split