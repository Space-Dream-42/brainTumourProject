from cProfile import label
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
data_dir = os.path.join(dataset_dir, "cropped")

train_images_dir = os.path.join(data_dir, "imagesTr")
train_labels_dir = os.path.join(data_dir, "labelsTr")

test_images_dir = os.path.join(data_dir, "imagesTs")
test_labels_dir = os.path.join(data_dir, "labelsTs")


def delete_folder(folder_path):
    """
    Deletes a folder at the specified path.
    """
    file_paths = glob.glob(os.path.join(folder_path, "*"))

    for f in file_paths:
        os.remove(f)
    
    shutil.rmtree(folder_path)


def delete_imagesTs():
    """
    Deletes the folder with extracted test images.
    """
    delete_folder(os.path.join(dataset_dir, "imagesTs"))


def folder_structure_exits():
    """
    Returns True if the Data direcory exists, else False.
    """
    return os.path.exists(os.path.join(parent_dir, 'Data'))
    
    
def create_folder_structure():
    """
    Creates all necessary directorys to set up the project. 
    """
    os.mkdir(data_dir)
    os.mkdir(train_images_dir)
    os.mkdir(train_labels_dir)
    os.mkdir(test_images_dir)
    os.mkdir(test_labels_dir)


def get_numpy_arr_of_nii_file(path):
    """
    Returns a given nii file as a numpy file.
    """
    sitk_arr = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(sitk_arr)
   
    
def crop_image_arr(arr):
    """
    Returns a cropped version of the given 3d input cube to reduce
    the large amount of background.  
    """
    return arr[:, :, 19:211, 19:211]


def crop_label_arr(arr):
    """
    Returns a cropped version of the given 3d label to reduce
    the large amount of background.  
    """
    return arr[:,19:211, 19:211]


def delete_original_dataset():
    """
    Deletes the extracted dataset.
    """
    json_file_path = os.path.join(dataset_dir, "dataset.json")
    labels_path = os.path.join(dataset_dir, "labelsTr")
    images_path = os.path.join(dataset_dir, "imagesTr")
    
    delete_folder(labels_path)
    delete_folder(images_path)
    os.remove(json_file_path)


def get_file_names():
    """
    Returns a list of all training file names.
    """
    with open(os.path.join(dataset_dir,'dataset.json')) as json_file:
        data = json.load(json_file)
        train_filenames = data['training']
    
    return train_filenames


def extract_crop_and_save_image_and_label_file(image_path_relative, label_path_relative, file_name, train):
    """
    Saves the cropped version of a given pair of image and label files.
    """
    if train:
        images_dir = train_images_dir
        labels_dir = train_labels_dir
    else:
        images_dir = test_images_dir
        labels_dir = test_labels_dir
    
    # extract and save image
    img_path = os.path.join(dataset_dir, image_path_relative)
    img_arr = crop_image_arr(get_numpy_arr_of_nii_file(img_path))
    img_arr_path = os.path.join(images_dir, file_name)
    np.save(img_arr_path, img_arr)

    # extract and save labels
    label_path = os.path.join(dataset_dir, label_path_relative)
    label_arr = crop_label_arr(get_numpy_arr_of_nii_file(label_path))
    label_arr_path = os.path.join(labels_dir, file_name)
    np.save(label_arr_path, label_arr)


def main():

    if os.path.exists(os.path.join(dataset_dir, "imagesTs")):
        print("Deleting the folder 'imagesTs'", end="\r")
        delete_imagesTs()
        print("Deleting the folder 'imagesTs' is done!")

    if not(folder_structure_exits()):
        print("Creating folder structure", end="\r")
        create_folder_structure()
        print("Creating the folder structure is done!")

    print("Extracting filenames of json-file.", end="\r")
    file_names = get_file_names()
    print("Extracting filenames of json-file is done!")

    # perform a train-test-split
    num_of_train_files = 400
    num_of_test_files = 84

    for i, img_path_dict in enumerate(file_names):
        image_path_gz = img_path_dict['image'][2:]
        label_path_gz = img_path_dict['label'][2:]

        if i < num_of_train_files:
            #train-files
            if i == num_of_train_files-1:
                print(f'Extracting training_file {i+1}/{num_of_train_files}')
            else:
                print(f'Extracting training_file {i+1}/{num_of_train_files}', end="\r")
            extract_crop_and_save_image_and_label_file(image_path_gz,label_path_gz,str(i),True)

        else:
            # test-files
            if i == num_of_test_files+num_of_train_files-1:
                print(f'Extracting test_file {i-num_of_train_files+1}/{num_of_test_files}')
            else:
                print(f'Extracting test_file {i-num_of_train_files+1}/{num_of_test_files}', end="\r")
            extract_crop_and_save_image_and_label_file(image_path_gz,label_path_gz,str(i-num_of_train_files),False)


    print("Extracting training and test files is done!")

    print("Deleting the original dataset.", end="\r")
    delete_original_dataset()
    print("Deleting the original dataset is done!")


if __name__ == "__main__":
    main()
