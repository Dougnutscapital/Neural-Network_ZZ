import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import load_model
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy

img_size = 224

def prepareImages(data, m, dataset):
    print("Preparing images")
    X_train = np.zeros((m, img_size, img_size, 3))
    count = 0

    for fig in data['Image']:
        img = image.load_img(dataset + "/" + fig, target_size=(img_size, img_size, 3))
        x = image.img_to_array(img)
        x = preprocess_input(x)

        X_train[count] = x
        if (count % 500 == 0):
            print("Processing image: ", count + 1, ", ", fig)
        count += 1

    return X_train

img_1 = './train/0a0c1df99.jpg'

def load_img(img_name):
    X = np.zeros((img_size, img_size, 3))
    img = image.load_img(img_name, target_size=(img_size, img_size, 3))
    X = image.img_to_array(img)
    X = preprocess_input(X)
    return X

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

model = load_model('white_model_new.h5',custom_objects={'top_5_accuracy': top_5_accuracy})

# print(model.summary())

result = model.predict(load_img(img_1).reshape(1,224,224,3))

print(result)

df_train = pd.read_csv('train.csv')
# df_train = df_train.head(5000)
print(df_train.head())

def prepare_labels(y):
    y = df_train['Id']
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    one_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    one_encoded = one_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = one_encoded
    # print(y.shape)
    return y, label_encoder, one_encoder

y, label_encoder, one_encoder = prepare_labels(df_train['Id'])

result_1 = one_encoder.inverse_transform(result)

result_2 = label_encoder.inverse_transform(result_1)


print(result_2)



