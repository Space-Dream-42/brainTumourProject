This is where your local dataset would live.

The subfolders should look something like this:
To get all of them, first download the compressed dataset
and then execute the Data_Extraction notebook to get the 'extracted' - directory.


    Task01_BrainTumour              (Dataset)
       |_ dataset.json
       |_ imagesTr                  (here are the compressed nii-files)
       |_ labelsTr
       |_ imagesTs
       |_ extracted                 (here are the numpy files)
           |_ imagesTr
           |_ labelsTr
           |_ imagesTs
         
