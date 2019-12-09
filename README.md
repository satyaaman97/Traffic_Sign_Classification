# Traffic_Sign_Classification

## Team Member:
* David Young-Jae Kim
* SeungJoon Kim
* Hao Wu
* Aman Satya

# Data Source and Some Instructions to Get Started

Dataset source: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed

Necessary steps before starting processing data:

* Download file train.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#train.pickle
* Download file valid.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#valid.pickle
* Download file test.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#test.pickle
* Download file label_names.csv here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#label_names.csv

# Dependencies
* imblearn
* cv2
* tqdm
* scikit-learn
* keras
* tensorflow
* numpy
* matplotlib
* pickle

# Dataset Generation

## Main source
* src/datasets_preparing_org.py

## Parameters
* orgSet - If True, it generates an original dataset with the same format as the other generated dataset. Default is True.
* isGray - If True, it generates Grayscale datasets instead of RGB datasets. Default is False.
* useNormalize - If True, it applies normalization and standardization. Default is True.

## Running the source
Put all the pickle files and csv file in the same directory.

Run the datasets_preparing_org.py

Then it generates the original dataset, 3 oversampling datasets, and 4 undersampling datasets.

All datasets are normalized and standardized by default.

It could take some time to generate. So if you want to generate specific dataset seperately, you can comment out the last part of the source including "makeCustomSampling" with various dataset manipulation methods.

# Model Creation and Model Loading

## Get Datasets and Models directly online
For your convenient, all the balanced dataset and models are already created online. You can visit this [link](https://drive.google.com/drive/u/1/folders/1Qjp5h6Ir3IWQXK9jsujPb8MAXGakxYPL) to get all datasets and models.

## Create Model
Run the src/createModels.ipynb. Load the appropriate dataset and using whatever model you like. We provide a toy dataset which is ToySet_rgb.pickle to help you start.

## Load Model and Get ROC&PR curve
Run the src/loadModel.ipynb. You can load the model created in the previous step, and the corresponding dataset. After that, you can plot the ROC&PR curve on test data.

# Result Analysis and Our Poster
![Poster](Poster.pdf)