import numpy as np
import csv
import pickle
import cv2
import os

from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, merge, ZeroPadding2D, AveragePooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LRN_helper import LRN2D

number_of_classes = 2
dimension = 224
number_of_channels = 3

def load_features_dataset(directory, number_of_files):
    dataset_features = []

    for current_dir in os.walk(directory):
        for current_file_number in range(1, number_of_files+1):
            current_path_with_file = directory + "/" + str(current_file_number) + ".jpg"
            img = cv2.imread(current_path_with_file)
            img = img.reshape((dimension * dimension * number_of_channels))  # flatten
            dataset_features.append(img)

    return np.array(dataset_features)

def load_labels_dataset(file):
    ignore_first_line = True
    dataset_labels = []

    with open(file) as f:
        dataset_csv_reader = csv.reader(f, delimiter=",")
        for line in dataset_csv_reader:
            if ignore_first_line:
                ignore_first_line = False

            else:
                label = np.zeros(2, dtype=int)
                label[int(line[1])] = 1
                dataset_labels.append(label)

    return np.array(dataset_labels)

def inception_v3():
    input_tensor = Input(shape=(dimension, dimension, number_of_channels))
    model = InceptionV3(input_tensor=input_tensor, weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x = Dense(number_of_classes, activation='softmax')(model.output)
    model = Model(model.input, x)

    # the first 24 layers are not trained
    for layer in model.layers[:24]:
        layer.trainable = False

    return model

def Xception_model():
    input_tensor = Input(shape=(dimension, dimension, number_of_channels))
    model = Xception(input_tensor=input_tensor, weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x = Dense(number_of_classes, activation='softmax')(model.output)
    model = Model(model.input, x)

    # the first 24 layers are not trained
    for layer in model.layers[:24]:
        layer.trainable = False

    return model

def ResNet50_model():
    input_tensor = Input(shape=(dimension, dimension, number_of_channels))
    model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=True)
    model.layers.pop()
    model.outputs = [model.layers[-1].output]
    model.layers[-1].outbound_nodes = []
    x = Dense(number_of_classes, activation='softmax')(model.output)
    model = Model(model.input, x)

    # the first 24 layers are not trained
    for layer in model.layers[:24]:
        layer.trainable = False

    return model

dataset_test_features = load_features_dataset('G:/Sahil/MS in US/ASU/CRS Lab/InvasiveSpecies/test', 1531)
dataset_test_features = dataset_test_features / 255.0
# reshaping
dataset_test_features = dataset_test_features.reshape((-1, dimension, dimension, number_of_channels))
print('dataset_test_features.shape:', dataset_test_features.shape)

model = inception_v3()
model.load_weights('C:/Users/SAHIL/Desktop/run3/inceptionv3/weights-improvement-20-0.99.h5py')
predictions_inception_v3 = model.predict(dataset_test_features)
classes_inception_v3 = []
for arr in predictions_inception_v3:
    classes_inception_v3.append(np.argmax(arr))
classes_inception_v3 = np.array(classes_inception_v3)

model = Xception_model()
model.load_weights('C:/Users/SAHIL/Desktop/run3/xception/weights-improvement-22-0.98.h5py')
predictions_xception = model.predict(dataset_test_features)
classes_xception = []
for arr in predictions_xception:
    classes_xception.append(np.argmax(arr))
classes_xception = np.array(classes_xception)

model = ResNet50_model()
model.load_weights('C:/Users/SAHIL/Desktop/run3/resnet/weights-improvement-10-0.98.h5py')
predictions_resnet = model.predict(dataset_test_features)
classes_resnet = []
for arr in predictions_resnet:
    classes_resnet.append(np.argmax(arr))
classes_resnet = np.array(classes_resnet)

predictions = []
for i in range(0, predictions_inception_v3.shape[0]):
    if classes_inception_v3[i] == 0 and classes_xception[i] == 0 and classes_resnet[i] == 0:
        predictions.append(min(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 1 and classes_xception[i] == 1 and classes_resnet[i] == 1:
        predictions.append(max(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 0 and classes_xception[i] == 0 and classes_resnet[i] == 1:
        predictions.append(min(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 0 and classes_xception[i] == 1 and classes_resnet[i] == 0:
        predictions.append(min(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 1 and classes_xception[i] == 0 and classes_resnet[i] == 0:
        predictions.append(min(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 1 and classes_xception[i] == 1 and classes_resnet[i] == 0:
        predictions.append(max(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 1 and classes_xception[i] == 0 and classes_resnet[i] == 1:
        predictions.append(max(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    elif classes_inception_v3[i] == 0 and classes_xception[i] == 1 and classes_resnet[i] == 1:
        predictions.append(max(predictions_inception_v3[i][1], predictions_xception[i][1], predictions_resnet[i][1]))

    else:
        print('error')

predictions = np.array(predictions)
print(predictions)

# np.savetxt("FILENAME.csv", predictions, delimiter=",")
with open('ensemble.csv','w') as file:
    file.write('name,invasive')
    file.write('\n')
    for i in range(0, predictions.shape[0]):
        file.write(str(i+1))
        file.write(',')
        if 'e-05' in str(predictions[i]):
            file.write('0.0')
        else:
            file.write(str(predictions[i]))
        file.write('\n')