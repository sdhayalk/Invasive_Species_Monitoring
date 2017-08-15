import numpy as np
import csv
import pickle
import cv2
import os

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, merge, ZeroPadding2D, AveragePooling2D, Flatten, Dropout, Dense, Activation, BatchNormalization
from keras.regularizers import l2
from keras.models import Model, model_from_json
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from LRN_helper import LRN2D

number_of_classes = 2
dimension = 224
number_of_channels = 3
batch_size = 50
epochs = 50

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

    lrate = 0.001
    decay = 0.000001
    adam = Adam(lr=lrate, decay=decay)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    print(model.summary())
    return model

dataset_train_features = load_features_dataset('/input/train', 2295)
dataset_train_labels = load_labels_dataset('/input/train_labels.csv')

dataset_train_features = dataset_train_features / 255.0
# dataset_test_features = dataset_test_features / 255.0

# pickle_dump(dataset_train_features, 'dataset_train_features.pickle')
# pickle_dump(dataset_train_labels, 'dataset_train_labels.pickle')
#pickle_dump(dataset_test_features, 'dataset_test_features.pickle')
#pickle_dump(dataset_test_labels, 'dataset_test_labels.pickle')
#
# dataset_train_features = pickle_retrieve('dataset_train_features.pickle')
# dataset_train_labels = pickle_retrieve('dataset_train_labels.pickle')
# dataset_test_features = pickle_retrieve('dataset_test_features.pickle')
# dataset_test_labels = pickle_retrieve('dataset_test_labels.pickle')
#

# divide
dataset_test_features = dataset_train_features[2000:dataset_train_features.shape[0], :]
dataset_test_labels = dataset_train_labels[2000:dataset_train_labels.shape[0], :]
dataset_train_features = dataset_train_features[0:2000, :]
dataset_train_labels = dataset_train_labels[0:2000, :]

# append 90 degree images
dataset_train_features_temp = load_features_dataset('/input/train_90', 2295)
dataset_train_labels_temp = load_labels_dataset('/input/train_labels.csv')
dataset_train_features_temp = dataset_train_features_temp[0:2000, :]
dataset_train_labels_temp = dataset_train_labels_temp[0:2000, :]
dataset_train_features_temp = dataset_train_features_temp / 255.0
dataset_train_features = np.concatenate((dataset_train_features, dataset_train_features_temp), axis=0)
dataset_train_labels = np.concatenate((dataset_train_labels, dataset_train_labels_temp), axis=0)

# append 180 degree images
dataset_train_features_temp = load_features_dataset('/input/train_180', 2295)
dataset_train_labels_temp = load_labels_dataset('/input/train_labels.csv')
dataset_train_features_temp = dataset_train_features_temp[0:2000, :]
dataset_train_labels_temp = dataset_train_labels_temp[0:2000, :]
dataset_train_features_temp = dataset_train_features_temp / 255.0
dataset_train_features = np.concatenate((dataset_train_features, dataset_train_features_temp), axis=0)
dataset_train_labels = np.concatenate((dataset_train_labels, dataset_train_labels_temp), axis=0)

# append 270 degree images
dataset_train_features_temp = load_features_dataset('/input/train_270', 2295)
dataset_train_labels_temp = load_labels_dataset('/input/train_labels.csv')
dataset_train_features_temp = dataset_train_features_temp[0:2000, :]
dataset_train_labels_temp = dataset_train_labels_temp[0:2000, :]
dataset_train_features_temp = dataset_train_features_temp / 255.0
dataset_train_features = np.concatenate((dataset_train_features, dataset_train_features_temp), axis=0)
dataset_train_labels = np.concatenate((dataset_train_labels, dataset_train_labels_temp), axis=0)

# append flip images
dataset_train_features_temp = load_features_dataset('/input/train_flip', 2295)
dataset_train_labels_temp = load_labels_dataset('/input/train_labels.csv')
dataset_train_features_temp = dataset_train_features_temp[0:2000, :]
dataset_train_labels_temp = dataset_train_labels_temp[0:2000, :]
dataset_train_features_temp = dataset_train_features_temp / 255.0
dataset_train_features = np.concatenate((dataset_train_features, dataset_train_features_temp), axis=0)
dataset_train_labels = np.concatenate((dataset_train_labels, dataset_train_labels_temp), axis=0)

# reshaping
dataset_train_features = dataset_train_features.reshape((-1, dimension, dimension, number_of_channels))
dataset_test_features = dataset_test_features.reshape((-1, dimension, dimension, number_of_channels))
print('dataset_test_features.shape:', dataset_test_features.shape)
print('dataset_test_labels.shape:', dataset_test_labels.shape)

filepath = "/output/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model = inception_v3()
model.fit(dataset_train_features, dataset_train_labels, validation_data=(dataset_test_features, dataset_test_labels), epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
predictions = model.predict(dataset_test_features)

model_json = model.to_json()
with open("/output/model_g_1.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("/output/model_g_1.h5")
print("Saved model to disk")
