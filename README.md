# Traffic_Sign_Classification

Dataset source: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed

Necessary steps before starting processing data:

* Download file train.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#train.pickle
* Download file valid.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#valid.pickle
* Download file test.pickle here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#test.pickle
* Download file label_names.csv here: https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed#label_names.csv


# Dataset Generation

## Prerequisites
* imblearn https://imbalanced-learn.readthedocs.io/en/stable/index.html
* cv2 https://pypi.org/project/opencv-python/
* tqdm https://github.com/tqdm/tqdm

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
