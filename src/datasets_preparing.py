# Preparing datasets for further using
# Loading Traffic Signs and plotting all classes with their labels
# Plotting histogram with number of images for every class
# Equalizing training dataset making examples in the classes equal
# Preprocessing datasets
#   data0.pickle - Shuffling
#   data1.pickle - Shuffling, /255.0 Normalization
#   data2.pickle - Shuffling, /255.0 + Mean Normalization
#   data3.pickle - Shuffling, /255.0 + Mean + STD Normalization
#   data4.pickle - Grayscale, Shuffling
#   data5.pickle - Grayscale, Shuffling, Local Histogram Equalization
#   data6.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 Normalization
#   data7.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean Normalization
#   data8.pickle - Grayscale, Shuffling, Local Histogram Equalization, /255.0 + Mean + STD Normalization
# Saving preprocessed datasets into files


"""Importing library for object serialization
which we'll use for saving and loading serialized models"""
import pickle

# Importing other standard libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
#from pylab import text
import csv
#from PIL import Image
#from skimage.transform import resize
from collections import Counter

useNormalize = False

# Defining function for loading dataset from 'pickle' file
def load_rgb_data(file):
    # Opening 'pickle' file and getting images
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is needed to divide float by float when applying Normalization
        x = d['features'].astype(np.float32)   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        y = d['labels']                        # 1D numpy.ndarray type, for train = (34799,)
        s = d['sizes']                         # 2D numpy.ndarray type, for train = (34799, 2)
        c = d['coords']                        # 2D numpy.ndarray type, for train = (34799, 4)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
            'sizes'    - is a 2D array containing arrays (width, height),
                         representing the original width and height of the image.
            'coords'   - is a 2D array containing arrays (x1, y1, x2, y2),
                         representing coordinates of a bounding frame around the image.
        """

    # Returning ready data
    return x, y, s, c


# Defining function for converting data to grayscale
def rgb_to_gray_data(x_data):
    # Preparing zero valued array for storing GrayScale images with only one channel
    x_g = np.zeros((x_data.shape[0], 1, 32, 32))

    # Converting RGB images into GrayScale images
    # Using formula:
    # Y' = 0.299 R + 0.587 G + 0.114 B
    x_g[:, 0, :, :] = x_data[:, 0, :, :] * 0.299 + x_data[:, 1, :, :] * 0.587 + x_data[:, 2, :, :] * 0.114

    # Also, possible to do with OpenCV
    # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Returning ready data
    return x_g


# Defining function for getting texts for every class - labels
def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []

    # Opening 'csv' file and getting image's labels
    with open(file, 'r') as f:
        reader = csv.reader(f)
        # Going through all rows
        for row in reader:
            # Adding from every row second column with name of the label
            label_list.append(row[1])
        # Deleting the first element of list because it is the name of the column
        del label_list[0]
    # Returning resulted list
    return label_list


"""
https://www.rapidtables.com/convert/color/rgb-to-hsv.html
https://ru.wikipedia.org/wiki/HSV_(цветовая_модель)
"""


# Defining function for changing brightness
def brightness_changing(image):
    # Converting firstly image from RGB to HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # Defining random value for changing brightness
    random_brightness = 0.25 + np.random.uniform()
    # Implementing changing of Value channel of HSV image
    image_hsv[:, :, 2] = image_hsv[:, :, 2] * random_brightness
    # Converting HSV changed image to RGB
    image_rgb = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2RGB)
    # Returning image with changed brightness
    return image_rgb


"""
To rotate an image using OpenCV Python,
first, calculate the affine matrix that does the affine transformation (linear mapping of pixels),
then warp the input image with the affine matrix.

Example:

M = cv2.getRotationMatrix2D(center, angle, scale)
rotated = cv2.warpAffine(img, M, (w, h))

where

center:  center of the image (the point about which rotation has to happen)
angle:   angle by which image has to be rotated in the anti-clockwise direction
scale:   1.0 mean, the shape is preserved. Other value scales the image by the value provided
rotated: ndarray that holds the rotated image data

Note: Observe that the dimensions of the resulting image are provided same as that of the original image.
When we are rotating by 90 or 270 and would to affect the height and width as well,
swap height with width and width with height.

https://www.tutorialkart.com/opencv/python/opencv-python-rotate-image/
"""


# Defining function for changing rotation of image
def rotation_changing(image):
    # Defining angle range
    angle_range = 25
    # Defining angle rotation
    angle_rotation = np.random.uniform(angle_range) - angle_range / 2
    # Getting shape of image
    rows, columns, channels = image.shape
    # Implementing rotation
    # Calculating Affine Matrix
    affine_matrix = cv2.getRotationMatrix2D((columns / 2, rows / 2), angle_rotation, 1)
    # Warping original image with Affine Matrix
    rotated_image = cv2.warpAffine(image, affine_matrix, (columns, rows))
    # Returning rotated image
    return rotated_image


# Defining function for transformation: brightness + rotation
def transformation_brightness_rotation(image):
    return brightness_changing(rotation_changing(image))


# Defining function for getting random image of one label
def random_image(x_train, y_train, y_number):
    # Getting indexes of needed 'y_number' from 'y_train'
    # Defining True - False array
    image_indexes = np.where(y_train == y_number)
    # Getting random index of needed label
    # 'np.bincount(y_train)' - array with number of examples for every label
    # 'np.bincount(y_train)[y_number] - 1' - number of examples for 'y_number' label
    random_index = np.random.randint(0, np.bincount(y_train)[y_number] - 1)
    # Returning random image from 'x_train'
    # 'x_train[image_indexes]' - returns array with only 'y_number' label
    # 'x_train[image_indexes][random_index]' - random image of needed label
    return x_train[image_indexes][random_index]


# Defining function for equalization training dataset
def equalize_training_dataset(x_train, y_train):
    # Getting number of examples for every label
    number_of_examples_for_every_label = np.bincount(y_train)
    # Calculating total amount of unique labels
    number_of_labels = np.arange(len(number_of_examples_for_every_label))

    # Iterating over all number of labels
    # Showing progress ber with 'tqdm'
    for i in tqdm(number_of_labels):
        # Calculating how many examples is needed to add for current label
        # 'np.mean(number_of_examples_for_every_label)' - average number over examples for every label
        number_of_examples_to_add = int(np.mean(number_of_examples_for_every_label) * 2.5) - \
                                    number_of_examples_for_every_label[i]

        # Defining temporary arrays for collecting new images
        x_temp = []
        y_temp = []

        # Getting random image from current label
        # Transforming it and adding to the temporary arrays
        for j in range(number_of_examples_to_add):
            getting_random_image = random_image(x_train, y_train, i)
            x_temp.append(transformation_brightness_rotation(getting_random_image))
            y_temp.append(i)

        x_train = np.append(x_train, np.array(x_temp), axis=0)
        y_train = np.append(y_train, np.array(y_temp), axis=0)

    return x_train, y_train


# Defining function for Local Histogram Equalization
def local_histogram_equalization(image):
    # Creating CLAHE object with arguments
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))

    # Applying Local Histogram Equalization and returning resulted image
    return clahe.apply(image)


# Defining function for preprocessing loaded data
def preprocess_data(d, shuffle=False, lhe=False, norm_255=False, mean_norm=False, std_norm=False,
                    transpose=True, colour='rgb'):
    # Applying Shuffling
    if shuffle:
        # Shuffle data
        # Multi-dimensional arrays are only shuffled along the first axis
        # By using seed we generate two times the same random numbers
        # And save appropriate connection: image --> label
        np.random.seed(0)
        np.random.shuffle(d['x_train'])
        np.random.seed(0)
        np.random.shuffle(d['y_train'])
        np.random.seed(0)
        np.random.shuffle(d['x_validation'])
        np.random.seed(0)
        np.random.shuffle(d['y_validation'])
        np.random.seed(0)
        np.random.shuffle(d['x_test'])
        np.random.seed(0)
        np.random.shuffle(d['y_test'])
        # Also, possible to do like following:
        # x_train, y_train = shuffle(x_train, y_train)
        # This function is from sklearn library:
        # from sklearn.utils import shuffle

    # Applying Local Histogram Equalization
    if lhe and colour != 'rgb':
        # Function map applies first argument to all elements of the second argument
        # First argument in our case is a function
        # Second argument in our case is np array
        # We need to slice it in order to pass into the function only (32, 32) and not (1, 32, 32)
        # Also, map functions applies to first argument all images of the second argument
        # In our case it is a number of d['x_train'].shape[0]
        # Result we wrap with list and then list convert to np.array
        # And reshaping it to make it again 4D tensor

        d['x_train'] = list(map(local_histogram_equalization, d['x_train'][:, 0, :, :].astype(np.uint8)))
        d['x_train'] = np.array(d['x_train'])
        d['x_train'] = d['x_train'].reshape(d['x_train'].shape[0], 1, 32, 32)
        d['x_train'] = d['x_train'].astype(np.float32)
        d['x_validation'] = list(map(local_histogram_equalization, d['x_validation'][:, 0, :, :].astype(np.uint8)))
        d['x_validation'] = np.array(d['x_validation'])
        d['x_validation'] = d['x_validation'].reshape(d['x_validation'].shape[0], 1, 32, 32)
        d['x_validation'] = d['x_validation'].astype(np.float32)
        d['x_test'] = list(map(local_histogram_equalization, d['x_test'][:, 0, :, :].astype(np.uint8)))
        d['x_test'] = np.array(d['x_test'])
        d['x_test'] = d['x_test'].reshape(d['x_test'].shape[0], 1, 32, 32)
        d['x_test'] = d['x_test'].astype(np.float32)

    # Applying /255.0 Normalization
    if norm_255:
        # Normalizing whole data by dividing /255.0
        d['x_train'] = d['x_train'].astype(np.float32) / 255.0
        d['x_validation'] = d['x_validation'].astype(np.float32) / 255.0
        d['x_test'] = d['x_test'].astype(np.float32) / 255.0

        # Preparing 'mean image'
        # Subtracting the dataset by 'mean image' serves to center the data
        # It helps for each feature to have a similar range and gradients don't go out of control.
        # Calculating 'mean image' from training dataset along the rows by specifying 'axis=0'
        # We CALCULATE 'mean image' ONLY FROM TRAINING dataset
        # Calculating mean image from training dataset along the rows by specifying 'axis=0'
        mean_image = np.mean(d['x_train'], axis=0)  # numpy.ndarray (3, 32, 32)
        # Saving calculated 'mean_image' into 'pickle' file
        # We will use it when preprocess input data for classifying
        # We will need to subtract input image for classifying
        # As we're doing now for training, validation and testing data
        dictionary = {'mean_image_' + colour: mean_image}
        with open('mean_image_' + colour + '.pickle', 'wb') as f_mean_image:
            pickle.dump(dictionary, f_mean_image)

        # Preparing 'std image'
        # Calculating standard deviation from training dataset along the rows by specifying 'axis=0'
        std = np.std(d['x_train'], axis=0)  # numpy.ndarray (3, 32, 32)
        # Saving calculated 'std' into 'pickle' file
        # We will use it when preprocess input data for classifying
        # We will need to divide input image for classifying
        # As we're doing now for training, validation and testing data
        dictionary = {'std_' + colour: std}
        with open('std_' + colour + '.pickle', 'wb') as f_std:
            pickle.dump(dictionary, f_std)

    # Applying Mean Normalization
    if mean_norm:
        # Normalizing data by subtracting with 'mean image'
        # Getting saved data for 'mean image'
        # Opening file for reading in binary mode
        with open('mean_image_' + colour + '.pickle', 'rb') as f:
            mean_image = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

        d['x_train'] -= mean_image['mean_image_' + colour]
        d['x_validation'] -= mean_image['mean_image_' + colour]
        d['x_test'] -= mean_image['mean_image_' + colour]

    # Applying STD Normalization
    if std_norm:
        # Normalizing data by dividing with 'standard deviation'
        # Getting saved data for 'std image'
        # Opening file for reading in binary mode
        with open('std_' + colour + '.pickle', 'rb') as f:
            std = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

        # Don't forget to change names for mean and std files when preprocessing for grayscale purposes
        d['x_train'] /= std['std_' + colour]
        d['x_validation'] /= std['std_' + colour]
        d['x_test'] /= std['std_' + colour]

    # WARNING!
    # Do not make transpose starting from data1
    # As data0 was already transposed
    if transpose:
        # Transposing every dataset to make channels come first
        d['x_train'] = d['x_train'].transpose(0, 3, 1, 2)  # (86989, 3, 32, 32)
        d['x_validation'] = d['x_validation'].transpose(0, 3, 1, 2)  # (86989, 3, 32, 32)
        d['x_test'] = d['x_test'].transpose(0, 3, 1, 2)  # (86989, 3, 32, 32)

    # Returning preprocessed data
    return d


# WARNING! Load and preprocess data for rgb and grayscale separately


# <---------->
# Option 1 - rgb data --> starts here
#
# # Loading rgb data from training dataset
# x_train, y_train, s_train, c_train = load_rgb_data('train.pickle')
#
# # Loading rgb data from validation dataset
# x_validation, y_validation, s_validation, c_validation = load_rgb_data('valid.pickle')
#
# # Loading rgb data from test dataset
# x_test, y_test, s_test, c_test = load_rgb_data('test.pickle')
#
# # Getting texts for every class
# label_list = label_text('label_names.csv')
#
# # Plotting 43 unique examples with their label's names
# # And histogram of 43 classes with their number of examples
# plot_unique_examples(x_train, y_train)
#
# # Plotting 43 good quality examples to show in GUI for driver
# plot_signs()
#
# # Implementing equalization of training dataset
# x_train, y_train = equalize_training_dataset(x_train.astype(np.uint8), y_train)
#
# # Plotting 43 unique examples with their label's names
# # And histogram of 43 classes with their number of examples
# plot_unique_examples(x_train, y_train)
#
# # Putting loaded and equalized data into the dictionary
# # Equalization is done only for training dataset
# d_loaded = {'x_train': x_train, 'y_train': y_train,
#             'x_validation': x_validation, 'y_validation': y_validation,
#             'x_test': x_test, 'y_test': y_test,
#             'labels': label_list}


# WARNING! It is important to run different preprocessing approaches separately
# Otherwise, dictionary will change values increasingly
# Also, creating separate dictionaries like 'd0, d1, d2, d3' will not help
# As they all contain same references to the datasets


# # Applying preprocessing
# data0 = preprocess_data(d_loaded, shuffle=True, transpose=True)
# print('Before Backward Calculation')
# print(data0['x_train'][0, 0, :, 0])
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data0.pickle', 'wb') as f:
#     pickle.dump(data0, f)
# # Releasing memory
# del data0

# # Applying preprocessing
# # Loading 'data0.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data0.pickle', 'rb') as f:
#     d_0_1 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data0 --> data1
# data1 = preprocess_data(d_0_1, shuffle=False, norm_255=True, transpose=False, colour='rgb')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data1.pickle', 'wb') as f:
#     pickle.dump(data1, f)
# # Releasing memory
# del d_0_1
# del data1

# # Applying preprocessing
# # Loading 'data0.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data0.pickle', 'rb') as f:
#     d_0_2 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data0 --> data2
# data2 = preprocess_data(d_0_2, shuffle=False, norm_255=True, mean_norm=True, transpose=False,
#                         colour='rgb')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data2.pickle', 'wb') as f:
#     pickle.dump(data2, f)
# # Releasing memory
# del d_0_2
# del data2

# # Applying preprocessing
# # Loading 'data0.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data0.pickle', 'rb') as f:
#     d_0_3 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data0 --> data3
# data3 = preprocess_data(d_0_3, shuffle=False, norm_255=True, mean_norm=True, std_norm=True,
#                         transpose=False, colour='rgb')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data3.pickle', 'wb') as f:
#     pickle.dump(data3, f)
# # Releasing memory
# del d_0_3
# del data3


# # Checking received preprocessed data by doing backward calculations
# # Getting mean and std
# # Opening file for reading in binary mode
# with open('mean_image_rgb.pickle', 'rb') as f:
#     mean_image_rgb = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
#
# # Opening file for reading in binary mode
# with open('std_rgb.pickle', 'rb') as f:
#     std_rgb = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
#
# # Loading 'data3.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data3.pickle', 'rb') as f:
#     data3 = pickle.load(f, encoding='latin1')  # dictionary type
#
# print(data3['x_train'].shape)
# # Starting from fully preprocessed dataset
# d3 = data3['x_train']
# print(d3.shape)  # (86989, 3, 32, 32)
#
# # Multiplying by std
# d2 = d3 * std_rgb['std_rgb']
# print(d2.shape)  # (86989, 3, 32, 32)
#
# # Adding with mean
# d1 = d2 + mean_image_rgb['mean_image_rgb']
# print(d1.shape)  # (86989, 3, 32, 32)
#
# # Multiplying by 255.0
# d0 = d1 * 255.0
# print(d0.shape)  # (86989, 3, 32, 32)
#
# # Showing result
# print('After Backward Calculation')
# print(d0[0, 0, :, 0])

# <---------->
# Option 1 - rgb data --> ends here


# <---------->
# Option 2 - grayscale data --> starts here

# # Loading 'data0.pickle' rgb dataset and going further with it
# # Opening file for reading in binary mode
# with open('data0.pickle', 'rb') as f:
#     data0 = pickle.load(f, encoding='latin1')  # dictionary type
#
# # Converting rgb data to grayscale for training dataset
# x_train = rgb_to_gray_data(data0['x_train'])
#
# # Converting rgb data to grayscale for validation dataset
# x_validation = rgb_to_gray_data(data0['x_validation'])
#
# # Converting rgb data to grayscale for testing dataset
# x_test = rgb_to_gray_data(data0['x_test'])
#
# # Putting loaded data into the dictionary
# d_loaded_gray = {'x_train': x_train, 'y_train': data0['y_train'],
#                  'x_validation': x_validation, 'y_validation': data0['y_validation'],
#                  'x_test': x_test, 'y_test': data0['y_test'],
#                  'labels': data0['labels']}
#
# # Showing the image by using obtained array with only one channel
# # Pay attention that when we use only one channeled array of image
# # We need to use (32, 32) and not (32, 32, 1) to show with 'plt.imshow'
# plt.imshow(x_train[9000, 0, :, :].astype(np.uint8), cmap=plt.get_cmap('gray'))
# plt.show()


# WARNING! It is important to run different preprocessing approaches separately
# Otherwise, dictionary will change values increasingly
# Also, creating separate dictionaries like 'd0, d1, d2, d3' will not help
# As they all contain same references to the datasets


# # Saving loaded and preprocessed data into 'pickle' file
# with open('data4.pickle', 'wb') as f:
#     pickle.dump(d_loaded_gray, f)
# # Releasing memory
# del d_loaded_gray

# # Applying preprocessing
# # Loading 'data4.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data4.pickle', 'rb') as f:
#     d_4_5 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data4 --> data5
# data5 = preprocess_data(d_4_5, shuffle=False, lhe=True, transpose=False, colour='gray')
# # Saving loaded and preprocessed data into 'pickle' file
# print('Before Backward Calculation')
# print(data5['x_train'][0, 0, :, 0])
# with open('data5.pickle', 'wb') as f:
#     pickle.dump(data5, f)
# # Releasing memory
# del d_4_5
# del data5

# # Applying preprocessing
# # Loading 'data4.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data4.pickle', 'rb') as f:
#     d_4_6 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data4 --> data6
# data6 = preprocess_data(d_4_6, shuffle=False, lhe=True, norm_255=True, transpose=False,
#                         colour='gray')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data6.pickle', 'wb') as f:
#     pickle.dump(data6, f)
# # Releasing memory
# del d_4_6
# del data6

# # Applying preprocessing
# # Loading 'data4.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data4.pickle', 'rb') as f:
#     d_4_7 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data4 --> data7
# data7 = preprocess_data(d_4_7, shuffle=False, lhe=True, norm_255=True, mean_norm=True,
#                         transpose=False, colour='gray')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data7.pickle', 'wb') as f:
#     pickle.dump(data7, f)
# # Releasing memory
# del d_4_7
# del data7

# # Applying preprocessing
# # Loading 'data4.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data4.pickle', 'rb') as f:
#     d_4_8 = pickle.load(f, encoding='latin1')  # dictionary type
# # Preprocessing data4 --> data8
# data8 = preprocess_data(d_4_8, shuffle=False, lhe=True, norm_255=True, mean_norm=True, std_norm=True,
#                         transpose=False, colour='gray')
# # Saving loaded and preprocessed data into 'pickle' file
# with open('data8.pickle', 'wb') as f:
#     pickle.dump(data8, f)
# # Releasing memory
# del d_4_8
# del data8


# # Checking received preprocessed data by doing backward calculations
# # Getting mean and std
# # Opening file for reading in binary mode
# with open('mean_image_gray.pickle', 'rb') as f:
#     mean_image_gray = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
#
# # Opening file for reading in binary mode
# with open('std_gray.pickle', 'rb') as f:
#     std_gray = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
#
# # Loading 'data8.pickle' dataset and going further with it
# # Opening file for reading in binary mode
# with open('data8.pickle', 'rb') as f:
#     data8 = pickle.load(f, encoding='latin1')  # dictionary type
#
# # Starting from fully preprocessed dataset
# d8 = data8['x_train']
# print(d8.shape)  # (86989, 1, 32, 32)
#
# # Multiplying by std
# d7 = d8 * std_gray['std_gray']
# print(d7.shape)  # (86989, 1, 32, 32)
#
# # Adding with mean
# d6 = d7 + mean_image_gray['mean_image_gray']
# print(d6.shape)  # (86989, 1, 32, 32)
#
# # Multiplying by 255.0
# d5 = d6 * 255.0
# print(d5.shape)  # (86989, 1, 32, 32)
#
# # Showing result
# print('After Backward Calculation')
# print(d5[0, 0, :, 0])

# <---------->
# Option 2 - grayscale data --> ends here


# <---------->
# Option 3 - Joon's grayscale data with SMOTE and ADASYN --> starts here

# --------------------
# oversampling methods
# --------------------
def equalize_training_dataset_with_SMOTE(x_train, y_train):
    from imblearn.over_sampling import SMOTE

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = SMOTE(sampling_strategy='not majority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_BorderlineSMOTE(x_train, y_train):
    from imblearn.over_sampling import BorderlineSMOTE

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = BorderlineSMOTE(sampling_strategy='not majority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_KMeansSMOTE(x_train, y_train):
    from imblearn.over_sampling import KMeansSMOTE

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = KMeansSMOTE(sampling_strategy='not majority', n_jobs=8, cluster_balance_threshold=0.009).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_SMOTENC(x_train, y_train):
    from imblearn.over_sampling import SMOTENC

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = SMOTENC(sampling_strategy='not majority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_SVMSMOTE(x_train, y_train):
    from imblearn.over_sampling import SVMSMOTE

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = SVMSMOTE(sampling_strategy='not majority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_ADASYN(x_train, y_train):
    from imblearn.over_sampling import ADASYN

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = ADASYN(sampling_strategy='not majority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

# ---------------------
# undersampling methods
# ---------------------
def equalize_training_dataset_with_ClusterCentroids(x_train, y_train):
    from imblearn.under_sampling import ClusterCentroids

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = ClusterCentroids(sampling_strategy='not minority', n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_CondensedNN(x_train, y_train):
    from imblearn.under_sampling import CondensedNearestNeighbour

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = CondensedNearestNeighbour(sampling_strategy={i : 180 for i in range(0, 43)}, n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
#    x_resampled, y_resampled = CondensedNearestNeighbour(sampling_strategy='not minority', n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_EditedNN(x_train, y_train):
    from imblearn.under_sampling import EditedNearestNeighbours

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = EditedNearestNeighbours(sampling_strategy='not minority', n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_RepeatedEditedNN(x_train, y_train):
    from imblearn.under_sampling import RepeatedEditedNearestNeighbours

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = RepeatedEditedNearestNeighbours(sampling_strategy='not minority', n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_AllKNN(x_train, y_train):
    from imblearn.under_sampling import AllKNN

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = AllKNN(sampling_strategy='not minority', n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_InstHardThres(x_train, y_train):
    from imblearn.under_sampling import InstanceHardnessThreshold

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = InstanceHardnessThreshold(sampling_strategy={i : 180 for i in range(0, 43)}, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_NearMiss(x_train, y_train):
    from imblearn.under_sampling import NearMiss

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = NearMiss(sampling_strategy={i : 180 for i in range(0, 43)}, version=1, n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_NClearningRule(x_train, y_train):
    from imblearn.under_sampling import NeighbourhoodCleaningRule

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = NeighbourhoodCleaningRule(sampling_strategy={i : 180 for i in range(0, 43)}, n_neighbors=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_OneSidedSel(x_train, y_train):
    from imblearn.under_sampling import OneSidedSelection

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = OneSidedSelection(sampling_strategy={i : 180 for i in range(0, 43)}, n_seeds_S=5, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_RandUnderSampler(x_train, y_train):
    from imblearn.under_sampling import RandomUnderSampler

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = RandomUnderSampler(sampling_strategy='not minority').fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

def equalize_training_dataset_with_TomekLinks(x_train, y_train):
    from imblearn.under_sampling import TomekLinks

    old_shape = list(x_train.shape)
    # reshape before using SMOTE
    x_tmp = np.reshape(x_train, (x_train.shape[0], -1))
    x_resampled, y_resampled = TomekLinks(sampling_strategy={i : 180 for i in range(0, 43)}, n_jobs=8).fit_resample(x_tmp, y_train)
    print(sorted(Counter(y_resampled).items()))
    # reshape after using SMOTE
    old_shape[0] = x_resampled.shape[0]
    x_resampled = np.reshape(x_resampled, tuple(old_shape))
    
    return x_resampled, y_resampled

# Defining function for converting data to grayscale
def rgb_to_gray_data2(x_data):
    # Preparing zero valued array for storing GrayScale images with only one channel
    x_g = np.zeros((x_data.shape[0], 32, 32, 1))

    # Converting RGB images into GrayScale images
    # Using formula:
    # Y' = 0.299 R + 0.587 G + 0.114 B
    x_g[:, :, :, 0] = x_data[:, :, :, 0] * 0.299 + x_data[:, :, :, 1] * 0.587 + x_data[:, :, :, 2] * 0.114

    # Also, possible to do with OpenCV
    # cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Returning ready data
    return x_g.transpose(0,3,1,2)

def save_gray(data, filename):
    # Converting rgb data to grayscale for training dataset
    x_train = rgb_to_gray_data2(data['x_train'])

    # Converting rgb data to grayscale for validation dataset
    x_validation = rgb_to_gray_data2(data['x_validation'])

    # Converting rgb data to grayscale for testing dataset
    x_test = rgb_to_gray_data2(data['x_test'])
    
    # Putting loaded data into the dictionary
    d_loaded_gray = {'x_train': x_train, 'y_train': data['y_train'],
                     'x_validation': x_validation, 'y_validation': data['y_validation'],
                     'x_test': x_test, 'y_test': data['y_test'],
                     'labels': data['labels']}
    
    # Applying preprocessing
    if useNormalize:
        data_after = preprocess_data(d_loaded_gray, shuffle=True, transpose=False, colour='gray', lhe=True, norm_255=True, mean_norm=True, std_norm=True)
    else:
        data_after = preprocess_data(d_loaded_gray, shuffle=True, transpose=False, colour='gray', lhe=True)
    
    # Showing the image by using obtained array with only one channel
    # Pay attention that when we use only one channeled array of image
    # We need to use (32, 32) and not (32, 32, 1) to show with 'plt.imshow'
    print('Sample Images')
    for i in range(10):
        plt.imshow(data_after['x_train'][i, 0, :, :].astype(np.uint8), cmap=plt.get_cmap('gray'))
        plt.show()
    print(data_after['x_train'][0,0,:,:])
    print('Final shape of dataset: ', data_after['x_train'].shape)
    
    # Saving loaded and preprocessed data into 'pickle' file
    with open(filename, 'wb') as f:
        pickle.dump(data_after, f)
    # Releasing memory
    del data_after

def save_rgb(data, filename):
    # Converting rgb data to grayscale for training dataset
    x_train = data['x_train'].transpose(0,3,1,2)

    # Converting rgb data to grayscale for validation dataset
    x_validation = data['x_validation'].transpose(0,3,1,2)

    # Converting rgb data to grayscale for testing dataset
    x_test = data['x_test'].transpose(0,3,1,2)
    
    # Putting loaded data into the dictionary
    d_loaded_rgb = {'x_train': x_train, 'y_train': data['y_train'],
                     'x_validation': x_validation, 'y_validation': data['y_validation'],
                     'x_test': x_test, 'y_test': data['y_test'],
                     'labels': data['labels']}
    
    # Applying preprocessing
    if useNormalize:
        data_after = preprocess_data(d_loaded_rgb, shuffle=True, transpose=False, lhe=False, norm_255=True, mean_norm=True, std_norm=True)
    else:
        data_after = preprocess_data(d_loaded_rgb, shuffle=True, transpose=False, lhe=False)
    
    # Showing the image by using obtained array with only one channel
    # Pay attention that when we use only one channeled array of image
    # We need to use (32, 32) and not (32, 32, 1) to show with 'plt.imshow'
    print('Sample Images')
    for i in range(10):
        tmp_img = data_after['x_train'][i].transpose(1,2,0).astype(np.uint8)
        plt.imshow(tmp_img)
        plt.show()
    print(data_after['x_train'][0,0,:,:])
    print('Final shape of dataset: ', data_after['x_train'].shape)
    
    # Saving loaded and preprocessed data into 'pickle' file
    with open(filename, 'wb') as f:
        pickle.dump(data_after, f)
    # Releasing memory
    del data_after

def makeCustomSampling(method, name):
    print('START ' + name + ' over/under sampling')
    # Loading rgb data from training dataset
    x_train, y_train, s_train, c_train = load_rgb_data('train.pickle')
    
    # Loading rgb data from validation dataset
    x_validation, y_validation, s_validation, c_validation = load_rgb_data('valid.pickle')
    
    # Loading rgb data from test dataset
    x_test, y_test, s_test, c_test = load_rgb_data('test.pickle')
    
    # Getting texts for every class
    label_list = label_text('label_names.csv')
    
    # origianl distribution
    print(sorted(Counter(y_train).items())) # 180~2110
    print(sorted(Counter(y_validation).items())) # 30~240
    print(sorted(Counter(y_test).items())) # 60~750

    # if toySet flag is true, make toy dataset with 540 samples with 3 classes
    if toySet:
        j = 0
        x_train2 = np.zeros((540,32,32,3), np.uint8)
        y_train2 = np.zeros(540, np.uint8)
        for i in range(y_train.shape[0]):
            if y_train[i] in (0,19,37): # classes that have 180 samples
                x_train2[j] = x_train[i]
                if y_train[i] == 0:
                    y_train2[j] = 0
                elif y_train[i] == 19:
                    y_train2[j] = 1
                elif y_train[i] == 37:
                    y_train2[j] = 2
                j += 1
        
        j = 0
        x_valid2 = np.zeros((90,32,32,3), np.uint8)
        y_valid2 = np.zeros(90, np.uint8)
        for i in range(y_validation.shape[0]):
            if y_validation[i] in (0,19,37): # classes that have 180 samples
                x_valid2[j] = x_validation[i]
                if y_validation[i] == 0:
                    y_valid2[j] = 0
                elif y_validation[i] == 19:
                    y_valid2[j] = 1
                elif y_validation[i] == 37:
                    y_valid2[j] = 2
                j += 1

        j = 0
        x_test2 = np.zeros((180,32,32,3), np.uint8)
        y_test2 = np.zeros(180, np.uint8)
        for i in range(y_test.shape[0]):
            if y_test[i] in (0,19,37): # classes that have 180 samples
                x_test2[j] = x_test[i]
                if y_test[i] == 0:
                    y_test2[j] = 0
                elif y_test[i] == 19:
                    y_test2[j] = 1
                elif y_test[i] == 37:
                    y_test2[j] = 2
                j += 1

                
        print('#ToySet#')
        x_train = x_train2
        y_train = y_train2
        x_validation = x_valid2
        y_validation= y_valid2
        x_test = x_test2
        y_test = y_test2
        print(sorted(Counter(y_train).items()))
        print(sorted(Counter(y_validation).items()))
        print(sorted(Counter(y_test).items()))

    
    # Implementing equalization of training dataset
    # use imblearn to equalize dataset
    x_new, y_new = method(x_train.astype(np.uint8), y_train)
    
    # Putting loaded and equalized data into the dictionary
    # Equalization is done only for training dataset
    d_new = {'x_train': x_new, 'y_train': y_new,
                'x_validation': x_validation, 'y_validation': y_validation,
                'x_test': x_test, 'y_test': y_test,
                'labels': label_list}
    
    if isGray:
        save_gray(d_new, name + '_gray.pickle')
    else:
        save_rgb(d_new, name + '_rgb.pickle')

    del d_new

def makeCustomSampling2(method, name):
    print('START ' + name + ' over/under sampling')
    # Loading rgb data from training dataset
    x_train, y_train, s_train, c_train = load_rgb_data('train.pickle')
    
    # Loading rgb data from validation dataset
    x_validation, y_validation, s_validation, c_validation = load_rgb_data('valid.pickle')
    
    # Loading rgb data from test dataset
    x_test, y_test, s_test, c_test = load_rgb_data('test.pickle')
    
    # Getting texts for every class
    label_list = label_text('label_names.csv')
    
    # origianl distribution
    print(sorted(Counter(y_train).items()))

    if isGray:
        name = name + '_gray'
        col = 'gray'
        # Converting rgb data to grayscale for training dataset
        x_train = rgb_to_gray_data2(x_train)
    
        # Converting rgb data to grayscale for validation dataset
        x_validation = rgb_to_gray_data2(x_validation)
    
        # Converting rgb data to grayscale for testing dataset
        x_test = rgb_to_gray_data2(x_test)
    else:
        name = name + '_rgb'
        col = 'rgb'
        # Converting rgb data to grayscale for training dataset
        x_train = x_train.transpose(0,3,1,2)
    
        # Converting rgb data to grayscale for validation dataset
        x_validation = x_validation.transpose(0,3,1,2)
    
        # Converting rgb data to grayscale for testing dataset
        x_test = x_test.transpose(0,3,1,2)

    d_gray = {'x_train': x_train, 'y_train': y_train,
                'x_validation': x_validation, 'y_validation': y_validation,
                'x_test': x_test, 'y_test': y_test,
                'labels': label_list}

    # Applying preprocessing
    if useNormalize:
        data_after = preprocess_data(d_gray, transpose=False, colour=col, lhe=isGray, norm_255=True, mean_norm=True, std_norm=True)
    else:
        data_after = preprocess_data(d_gray, transpose=False, colour=col, lhe=isGray)

    # Implementing equalization of training dataset
    # use imblearn to equalize dataset
    data_after['x_train'], data_after['y_train'] = method(data_after['x_train'].astype(np.uint8), data_after['y_train'])

    data_new = preprocess_data(data_after, shuffle=True, transpose=False)
    
    # Showing the image by using obtained array with only one channel
    # Pay attention that when we use only one channeled array of image
    # We need to use (32, 32) and not (32, 32, 1) to show with 'plt.imshow'
    print('Sample Images')
    
    for i in range(10):
        if isGray:
            plt.imshow(data_new['x_train'][i, 0, :, :].astype(np.uint8), cmap=plt.get_cmap('gray'))
        else:
            tmp_img = data_after['x_train'][i].transpose(1,2,0).astype(np.uint8)
            plt.imshow(tmp_img)
        plt.show()
    
    print('Final shape of dataset: ', data_new['x_train'].shape)
    
    # Saving loaded and preprocessed data into 'pickle' file
    with open(name, 'wb') as f:
        pickle.dump(data_new, f)
    # Releasing memory
    del data_new
    del data_after
    del d_gray

# normalize flag. if set it True, sample image is not correctly showed.
useNormalize = False
# toySet flag. if set it True, toySet will be generated.
toySet = False
# grayscale flag. if set it True, all image turns into the grayscale
isGray = False

if toySet:
    useNormalize = False
    makeCustomSampling(equalize_training_dataset_with_RandUnderSampler, 'ToySet1')
    useNormalize = True
    makeCustomSampling(equalize_training_dataset_with_RandUnderSampler, 'ToySet2')

# make oversampling with given methods
makeCustomSampling(equalize_training_dataset_with_SMOTE, 'SMOTE')
makeCustomSampling(equalize_training_dataset_with_BorderlineSMOTE, 'BorderlineSMOTE')
makeCustomSampling(equalize_training_dataset_with_KMeansSMOTE, 'KMeansSMOTE')

# cannot use SMOTENC because dataset has no categorical features
#makeCustomSampling(equalize_training_dataset_with_SMOTENC, 'SMOTENC')
# cannot use SVMSMOTE because it hangs on my computer even after an hour. don't know why...machine power is not enough??
#makeCustomSampling(equalize_training_dataset_with_SVMSMOTE, 'SVMSMOTE')
# cannot use ADASYN. I need more information about why it cannot be apllied to the dataset
#makeCustomSampling(equalize_training_dataset_with_ADASYN, 'ADASYN')

# make undersampling with given methods
makeCustomSampling(equalize_training_dataset_with_ClusterCentroids, 'ClusterCentroids')
makeCustomSampling(equalize_training_dataset_with_NearMiss, 'NearMiss')
makeCustomSampling(equalize_training_dataset_with_RandUnderSampler, 'RandUnderSampler')
# # of samples are almost same but not equal.
makeCustomSampling(equalize_training_dataset_with_InstHardThres, 'InstHardThres')


# those methods make # of samples not be equal to each other. So I skipped these methods.
#makeCustomSampling(equalize_training_dataset_with_CondensedNN, 'CondensedNN')
#makeCustomSampling(equalize_training_dataset_with_EditedNN, 'EditedNN')
#makeCustomSampling(equalize_training_dataset_with_RepeatedEditedNN, 'RepeatedEditedNN')
#makeCustomSampling(equalize_training_dataset_with_AllKNN, 'AllKNN')
#makeCustomSampling(equalize_training_dataset_with_NClearningRule, 'NClearningRule')
#makeCustomSampling(equalize_training_dataset_with_OneSidedSel, 'OneSidedSel')
#makeCustomSampling(equalize_training_dataset_with_TomekLinks, 'TomekLinks')

# makeCustomSampling2 applies normalization and standardization before over/under-sampling.
#makeCustomSampling2(equalize_training_dataset_with_SMOTE, 'SMOTE')

# <---------->
# Option 3 - Joon's grayscale data with SMOTE and ADASYN --> ends here