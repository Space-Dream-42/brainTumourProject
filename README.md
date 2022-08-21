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
4. Run "python3 setup_project.py" in your shell, to extract the nii-files to numpy file (Attention!: have in mind that you need at most 100GB of storage)

## Useful links:
[Github repo](https://github.com/Space-Dream-42/brainTumourProject) \
[Model files](https://drive.google.com/drive/folders/1pTMtH2817WEceukKP52Lep9QR-ZB2WKz?usp=sharing) \
[Our paper]()

## Used repos:
Please mention the github repos that you used for your solution.

## How to run the code:

### Where can i find what (abstract discription)?
- demo.ipynb: \
   Here we are executed our code, extracted results from it and visualized our results for our paper. Every discovery that we mention in our paper comes from here.
- custom_losses.py: \
   Here you can find custom programmed losses. In addition you can find get_loss to calculate automatically the loss for a provieded loss-function. 
- data_loading.py: \
   Here you can find the dataloader. The function get_train_test_iters gives you the data iterators. 
- dataset_utils.py: \
   Here you can find everything ragarding to manipulate and processing data.
- train.py: \
   Here you can find the train_model function to train a model.
- Optional: \
   Here we are experimenting with our code.
- Architectures: \
   Here you can find all model architectures that we build.

### Code documantion with example invocation

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