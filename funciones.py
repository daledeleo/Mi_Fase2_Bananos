# Importando las librerias a usar
import matplotlib.image as mpimg
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from tensorflow.python.keras import backend as K
from matplotlib import pyplot
import matplotlib.pyplot as plt
import pandas as pd
import os
import re
import numpy as np

from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras import Sequential, Model, models


# from tensorflow.keras.applications.inception_v3 import InceptionV3
# https://keras.io/api/applications/

from sklearn.model_selection import train_test_split
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
import keras
from keras.utils import to_categorical
import tensorflow as tf
from keras.constraints import maxnorm
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def cargar_dataset(images, directories, path):
    """
    Returns nothing.

            Parameters:
                    images (array): array empty
                    directories (array): array empty
                    dircount (array): array empty
                    path (string): path of dataset

            Returns:
                   nothing
    """
    dircount = []
    # CARGANDO EL DATASET
    dirname = os.path.join(os.getcwd(), path)  # nombre de la carpeta Raiz
    imgpath = dirname + os.sep

    prevRoot = ''
    cant = 0

    print("leyendo imagenes de ", imgpath)

    for root, dirnames, filenames in os.walk(imgpath):
        for filename in filenames:
            if re.search("\.(jpg|jpeg|png|bmp|tiff)$", filename):
                cant = cant+1
                filepath = os.path.join(root, filename)
                image = plt.imread(filepath)
                # image.resize((32, 32, 3))
                images.append(image)
                b = "Leyendo..." + str(cant)
                print(b, end="\r")
                if prevRoot != root:
                    print(root, cant)
                    prevRoot = root
                    directories.append(root)
                    dircount.append(cant)
                    cant = 0
    dircount.append(cant)

    dircount = dircount[1:]
    dircount[0] = dircount[0]+1
    print('Directorios leidos:', len(directories))
    # dict={directories[0]:}
    print("Imagenes en cada directorio", dircount)
    print('suma Total de imagenes en subdirs:', sum(dircount))
    return dircount


def agregar_etiquetas(directories, dircount, images):
    """
     Returns X, y.

             Parameters:
                     labels (array): array empty
                     tipos_banano (array): array empty
                     directories (array): array with directories of data
                     dircount (array): array with dircount of dataset
                     images (array): images of dataset

             Returns:
                    X num of data, y convert list a numpy
     """
    # Agregando etiquetas
    labels = []
    tipos_de_banano = []

    indice = 0
    for cantidad in dircount:
        for i in range(cantidad):
            labels.append(indice)
        indice = indice+1
    print("Cantidad etiquetas creadas: ", len(labels))

    tipos_de_banano = []
    indice = 0
    for directorio in directories:
        name = directorio.split(os.sep)
        print(indice, name[len(name)-1])
        tipos_de_banano.append(name[len(name)-1])
        indice = indice+1

    y = np.array(labels)
    X = np.array(images, dtype=np.uint8)  # convierto de lista a numpy
    return X, y


def normalizar_and_one_hot_encoding(X_data, y_data):
    """
    Returns X_data_one_hot.

            Parameters:
                    X_data (array): array with data to normalize
                    y_data (array): array with labels

            Returns:
                   X_data_one_hot data with one hot enconding
    """
    # normalizando las imagenes
    X_data = X_data.astype('float32')
    X_data = X_data / 255.

    # Change the labels from categorical to one-hot encoding
    Y_data_one_hot = to_categorical(y_data)
    return Y_data_one_hot

# load and prepare the image


def load_image(filename):
	# load the image
	img = load_img(filename, target_size=(224, 224))
	# convert to array
	img = img_to_array(img)
	# reshape into a single sample with 3 channels
	img = img.reshape(1, 224, 224, 3)
	# prepare pixel data
	img = img.astype('float32')
	img = img / 255.0
	return img


#Cantidad etiquetas creadas:  2403
#0 Class C
#1 Class A
#2 Class D
#3 Class B

# load an image and predict the class
def run_example(filename,path_modelo):
	# load the image
	img = load_image(filename)
	# load model
	model = load_model(path_modelo)
	# predict the class
	result = model.predict_classes(img)
	dic={
        0 :'Clase A',
        1: 'Clase B',
        2: 'Clase C',
        3 :'Clase D'}
	print(dic[result[0]])

'''
    # plot diagnostic learning curves
def summarize_diagnostics(history):
	# plot loss
	pyplot.subplot(211)
	pyplot.title('Cross Entropy Loss')
	pyplot.plot(history.history['loss'], color='blue', label='train')
	pyplot.plot(history.history['val_loss'], color='orange', label='test')
    pyplot.show()

	# plot accuracy
	pyplot.subplot(212)
	pyplot.title('Classification Accuracy')
	pyplot.plot(history.history['accuracy'], color='blue', label='train')
	pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    pyplot.show()

'''

print("Modulo importado con exito.... :)")