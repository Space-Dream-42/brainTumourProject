import numpy as np
import json
import os
import glob
import SimpleITK as sitk
import shutil

current_dir = os.getcwd()
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) 
dataset_dir = os.path.join(parent_dir, 'Task01_BrainTumour')

# data folder
data_dir = os.path.join(parent_dir, "data")

train_dir = os.path.join(data_dir, "train")
train_images_dir = os.path.join(train_dir, "images")
train_labels_dir = os.path.join(train_dir, "labels")

test_dir = os.path.join(data_dir, "test")
test_images_dir = os.path.join(test_dir, "images")
test_labels_dir = os.path.join(test_dir, "labels")


def delete_folder(folder_path):
    file_paths = glob.glob(os.path.join(folder_path, "*"))

    for f in file_paths:
        os.remove(f)
    
    shutil.rmtree(folder_path)

def delete_imagesTs():
    delete_folder(os.path.join(dataset_dir, "imagesTs"))

def folder_structure_exits():
    return os.path.exists(os.path.join(parent_dir, 'Data'))
    
def create_folder_structure():
    data_path = os.path.join(parent_dir, 'Data')
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")
    train_images_path = os.path.join(train_path, "images")
    train_labels_path = os.path.join(train_path, "labels")
    test_images_path = os.path.join(test_path, "images")
    test_labels_path = os.path.join(test_path, "labels")
    os.mkdir(data_path)
    os.mkdir(train_path)
    os.mkdir(test_path)
    os.mkdir(train_images_path)
    os.mkdir(train_labels_path)
    os.mkdir(test_images_path)
    os.mkdir(test_labels_path)

def get_numpy_arr_of_nii_file(path):
    sitk_arr = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_arr)
    
def crop_image_arr(arr):
    return arr[:, :, 19:211, 19:211]

def crop_label_arr(arr):
    return arr[:,19:211, 19:211]

def delete_original_dataset():
    json_file_path = os.path.join(dataset_dir, "dataset.json")
    labels_path = os.path.join(dataset_dir, "labelsTr")
    images_path = os.path.join(dataset_dir, "imagesTr")
    
    delete_folder(labels_path)
    delete_folder(images_path)
    os.remove(json_file_path)

def get_file_names():
    with open(dataset_dir + '/dataset.json') as json_file:
        data = json.load(json_file)
        train_filenames = data['training']
    
    return train_filenames

def main():
    try:
        if os.path.exists(os.path.join(dataset_dir, "imagesTs")):
            print("Deleting the folder 'imagesTs'", end="\r")
            delete_imagesTs()
            print("Deleting the folder 'imagesTs is done!'")
    except:
        print("Failed to delete the folder 'imagesTs'.")
        exit(1)

    try:
        if not(folder_structure_exits()):
            print("Creating folder structure", end="\r")
            create_folder_structure()
            print("Creating the folder structure is done!")
    except:
        print("Failed to create folder structure.")
        exit(1)

    try:
        print("Extracting filenames of json-file.", end="\r")
        file_names = get_file_names()
        print("Extracting filenames of json-file is done!")
    except:
        print("Failed to extract filenames of json-file.")
        exit(1)

    # 80/20 train-test-split
    num_of_files = len(file_names)
    num_of_train_files = 387
    num_of_test_files = 97

    # train-files
    for i, img_path_dict in enumerate(file_names):
        image_path_gz = img_path_dict['image'][2:]
        label_path_gz = img_path_dict['label'][2:]

        if i < num_of_test_files:
            print(f'Extracting training_file {i+1}/{num_of_train_files}', end="\r")

            # extract and save image
            img_path = os.path.join(dataset_dir, image_path_gz)
            img_arr = crop_image_arr(get_numpy_arr_of_nii_file(img_path))
            train_imag_path = os.path.join(train_images_dir, str(i))
            np.save(train_imag_path, img_arr)

            # extract and save labels
            label_path = os.path.join(dataset_dir, label_path_gz)
            label_arr = crop_label_arr(get_numpy_arr_of_nii_file(label_path))
            train_label_path = os.path.join(train_labels_dir, str(i))
            np.save(train_label_path, label_arr)

        else:
            # test-files
            print(f'Extracting test_file {i-385}/{num_of_train_files}', end="\r")
            # extract and save image
            img_path = os.path.join(dataset_dir, image_path_gz)
            img_arr = crop_image_arr(get_numpy_arr_of_nii_file(img_path))
            train_imag_path = os.path.join(test_images_dir, str(i))
            np.save(train_imag_path, img_arr)

            # extract and save labels
            label_path = os.path.join(dataset_dir, label_path_gz)
            label_arr = crop_label_arr(get_numpy_arr_of_nii_file(label_path))
            train_label_path = os.path.join(test_labels_dir, str(i))
            np.save(train_label_path, label_arr)


    try:
        print("Deleting the original dataset.", end="\r")
        delete_original_dataset()
        print("Deleting the original dataset is done!")
    except:
        print("Failed to delete the original dataset.")
        exit(1)


if __name__ == "__main__":
    main()
