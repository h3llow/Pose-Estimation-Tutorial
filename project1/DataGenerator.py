import pickle
import tensorflow as tf
import cv2
import os
import numpy as np
import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class DataGenerator(keras.utils.Sequence):
    def __init__(self, set, batch_size=32, shuffle=True):
        self.path = set + "/"
        self.subpath = "augmented/"
        if set in ["test", "validation"]:
            self.subpath = ""
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.counter = 0
        self.on_epoch_end()


    def __len__(self):
        """Required: Number of batches"""
        return int(np.ceil(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        """Generate and return one batch"""

        # Generate indexes of the batch
        indexes_batch = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        X = []
        y = []
        for index in indexes_batch:
            x_tmp, y_tmp = self.__data_generation(index)
            X.append(x_tmp)
            y.append(y_tmp)
        X = np.array(X)
        y = np.array(y)
        return X, y #tf.data.Dataset.from_tensor_slices((X, y))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = []
        self.counter += 1
        for label in [0, 1, 2, 3, 4]:
            for img_name in os.listdir(self.path + str(label) + "/" + self.subpath):
                self.indexes.append((self.path + str(label) + "/" + self.subpath + img_name, label))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _vectorize(self, j):
        e = np.zeros(5)
        e[j] = 1
        return e

    def __data_generation(self, index):
        img_path, label = index
        img = cv2.imread(img_path)
        if self.path == "test/":
            img = cv2.resize(img, (224, 224))
        else:
            label = self._vectorize(label)
        return img / 255, label
