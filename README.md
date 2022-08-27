# <div align="center">Brain Tumour Segmentation (BraTS) <br/> using an U-Net architecture</div>

## How to run our code
### Set up the project
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
3. Change into "Code" as a working directory
4. Run "python3 setup_project.py" in your shell, to extract the nii-files to numpy file (Attention!: have in mind that the dataset needs 70+ GB of storage)


### Additional ressources for our code
<details>
  <summary>High-level code overview (Click to expand)</summary>

### Code Overview
#### Notebooks
- demo.ipynb:<br/>
Here we are executed our code, extracted results from it and visualized our results for our paper. Every discovery that we mention in our paper comes from here.

#### Python-files
- custom_losses.py:<br/>
Here you can find custom programmed losses. In addition you can find get_loss to calculate automatically the loss for a provieded loss-function. 

- data_loading.py:<br/>
Here you can find the dataloader. The function get_train_test_iters gives you the data iterators. 

- dataset_utils.py:<br/>
Here you can find everything ragarding to manipulate and processing data.

- train.py:<br/>
Here you can find the train_model function to train a model.

#### Folders
- Optional:<br/>
Here we are experimenting with our code.

- Architectures:<br/>
Here you can find all model architectures that we build.
</details>


<details>
  <summary>File-structure figure (Click to expand)</summary>
<p align="left"><img src="https://github.com/Space-Dream-42/brainTumourProject/blob/main/images/filestructure.jpg?raw=true" width="700" height="500"></p>
</details>


<details>
  <summary>See our docs (Click to expand)</summary>
    
### Documentation with sample function calls
    
custom_losses.py: 
| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |

data_loading.py:
| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |

dataset_utils.py:
| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |

train.py:
| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |

Architectures
* unet_2d.py: 

| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |
* unet_3d.py: 

| Function/Class| Description | Example |
| ----------- | ----------- |----------- |
| Header      | Title       |Title       |
| Paragraph   | Text        |Text        |
</details>


## Useful links
[Our executive report]()  <br/>
[Github repo](https://github.com/Space-Dream-42/brainTumourProject) <br/>
[Model files](https://drive.google.com/drive/folders/1pTMtH2817WEceukKP52Lep9QR-ZB2WKz?usp=sharing) <br/>


## Acknowledgements
Thanks to Noel & Prof. Lippert <br/>
Please mention the github repos that you used for your solution.
