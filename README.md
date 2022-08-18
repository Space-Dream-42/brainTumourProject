# Brain Tumour Segmentation (BraTS)

## How to set up the project:
1. Download "Task01_BrainTumour.tar" under the website http://medicaldecathlon.com and extract the tar file.
2. Provide the following folder structure:
```
BT_Segmentation_Project
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
3. Install poetry, see https://python-poetry.org/docs/#installation 
4. Install the dependencies by executing `poetry install`
5. Switch to "Code" as a working directory
6. Run `poetry run python3 setup_project.py` in your shell, to extract the nii-files to numpy file (Attention!: have in mind that you need at least 70GB of storage)
7. If you want to save memory, you can delete "imagesTr", "labelsTr" and "imagesTs" in the folder "BT_Segmentation_Project/Task01_BrainTumour"
