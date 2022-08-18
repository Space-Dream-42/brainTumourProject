# Brain Tumour Segmentation (BraTS)

## How to set up the project:
1. Download "Task01_BrainTumour.tar" under the website http://medicaldecathlon.com and extract the tar file.
2. Provide the following folder structure:
```
BT_Segmentation_Project
    |_ README.md
    |_ Code
       |_ Data_Extraction.ipynb     (this file)
       |_ ...
    |_ Task01_BrainTumour           (Dataset)
       |_ dataset.json
       |_ imagesTr                  (here are the compressed nii-files)
       |_ labelsTr
       |_ imagesTs
    |_ Weights                      (trained model weights)
```
3. Swtich to "Code" as a working directory
4. Run "python3 setup_project.py" in your shell, to extract the nii-files to numpy file (Attention!: have in mind that you need at least 100GB of storage)

## How to run the code:
