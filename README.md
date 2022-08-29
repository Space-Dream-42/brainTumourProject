# <div align="center">Brain Tumour Segmentation (BraTS) <br/> using an U-Net architecture</div>

## How to run our code
### Set up the project
1. Download "Task01_BrainTumour.tar" under the [Medical Segmentation Decathlon](http://medicaldecathlon.com) website and extract the tar file.
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
4. Run "python3 setup_project.py" in your shell, to extract the nii-files to numpy file (Attention!: bear in mind that the dataset needs 70+ GB of storage)


### Additional ressources for our code
<details>
  <summary>High-level code overview (Click to expand)</summary>

### Code Overview
#### Notebooks
- demo.ipynb:<br/>
Here we present our model training along with evaluation metrics and animated visualizations. The majority of what we report in our paper is produced using this code.

#### Python-files
- custom_losses.py:<br/>
Contains custom programmed losses. In addition you can find the get_loss function to automatically calculate the loss for a provided loss-function. 

- data_loading.py:<br/>
This is the dataloader. The function get_train_test_iters gives you the data iterators. 

- dataset_utils.py:<br/>
Here we coded all functions for manipulating and processing data.

- train.py:<br/>
Here you can find the train_model function to train a model.

#### Folders
- Optional:<br/>
Here we stored additional experimental code that was not ultimately used in our trained models or demo.

- Architectures:<br/>
Here you can find all model architectures that we built.
</details>


<details>
  <summary>File-structure figure (Click to expand)</summary>-
<p align="left"><img src="https://github.com/Space-Dream-42/brainTumourProject/blob/main/images/filestructure.jpg?raw=true" width="700" height="500"></p>
</details>


## Useful links
[Our executive report](executive_report.pdf)  <br/>
[Github repo](https://github.com/Space-Dream-42/brainTumourProject) <br/>
[Model files](https://drive.google.com/drive/folders/1pTMtH2817WEceukKP52Lep9QR-ZB2WKz?usp=sharing) <br/>


## Acknowledgements
We would like to thank Prof. Dr. Chistoph Lippert, Tahir Miriyev, Noel Danz, Eshant English and Wei-Cheng Lai for sharing their experience and expertise with us. Special thanks to Noel Danz who gave insightful tips and comments to help guide the project.


## Used Github-repos
- Inspiration for our U-Net implementation: <br/>
https://github.com/Hsankesara/DeepResearch/blob/master/UNet/Unet.py <br/>

- Basis for adapted fuctions/classes, such as dice loss and focaltversky loss: <br/>
https://github.com/Mr-TalhaIlyas/Loss-Functions-Package-Tensorflow-Keras-PyTorch <br/>

- Reference was made to pytorch documentation throught the coding process: <br/>
https://pytorch.org/docs/stable/index.html
