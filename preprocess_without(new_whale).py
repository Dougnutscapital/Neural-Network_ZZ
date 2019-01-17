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
from keras.models import Model
import keras.backend as K
from keras.models import Sequential
import warnings

# data = pd.read_csv('train_processed.csv')
# unique = pd.value_counts(data.Id)
# print(unique.head())
# num_classes = unique.values.shape[0]
# print(unique.values)
# print(num_classes)



# df_train = pd.read_csv('train_processed.csv')
df_train = pd.read_csv('train_processed_1.csv')
# df_train = df.sample(frac=0.9,random_state=0)
# print(df_train.index)
# df_test = df.drop(df_train.index)
# print(df_test.shape)
# # df_train = df_train[df_train['Id']!='new_whale']
# # df_train = df_train.head(20000)
# print(df_train.head())
# print(df_train.shape)


# unique = pd.value_counts(df_train.Id)
# print(unique.head())
# num_classes = unique.values.shape[0]
# print(unique.values)
# print(num_classes)

img_size = 100


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

def prepare_labels(y):
    values = np.array(y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    # print(integer_encoded)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # print(onehot_encoded)

    y = onehot_encoded
    # print(y.shape)
    return y, label_encoder

# X = prepareImages(df_train, df_train.shape[0], "../train")
X = prepareImages(df_train, df_train.shape[0], "../home/lchn_guo/projects/WhalesServer/generated_train")
X /= 255

# from keras.preprocessing.image import ImageDataGenerator
#
# train_datagen = ImageDataGenerator(
#         rescale=1./255,
#         shear_range=0.2,
#         zoom_range=0.2,
#         horizontal_flip=True)
#
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# train_generator = train_datagen.flow_from_dataframe(dataframe = df_train,
#         directory='../home/lchn_guo/projects/WhalesServer/generated_train/',
#         x_col='Image', y_col='Id',
#         target_size=(224, 224),
#         batch_size=32,
#         )
#
# validation_generator = test_datagen.flow_from_dataframe(dataframe=df_test,
#         directory='../home/lchn_guo/projects/WhalesServer/generated_train/test/',
#         x_col='Image', y_col='Id',
#         target_size=(224, 224),
#         batch_size=32,
#         )

y, label_encoder = prepare_labels(df_train['Id'])

print(y.shape)

from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras import optimizers

nb_classes = 5005
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 50  # 冻结层的数量

# 添加新层
def add_new_last_layer(base_model, nb_classes):
  """
  添加最后的层
  输入
  base_model和分类数量
  输出
  新的keras的model
  """
  x = base_model.output
  x = GlobalAveragePooling2D()(x)
  x = Dense(FC_SIZE, activation='relu')(x) #new FC layer, random init
  predictions = Dense(nb_classes, activation='softmax')(x) #new softmax layer
  model = Model(input=base_model.input, output=predictions)
  return model

# 冻上base_model所有层，这样就可以正确获得bottleneck特征
def setup_to_transfer_learn(model, base_model):
  """Freeze all layers and compile the model"""
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
      layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
      layer.trainable = True


  # model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[top_5_accuracy])


def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# 定义网络框架
base_model = ResNet50(input_shape=(img_size, img_size, 3),weights='imagenet', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层
setup_to_transfer_learn(model, base_model)

# train_generator = train_datagen.flow_from_directory(
# train_dir,
# target_size=(IM_WIDTH, IM_HEIGHT),
# batch_size=batch_size,class_mode=’categorical’)


# def top_5_accuracy(y_true, y_pred):
#     return top_k_categorical_accuracy(y_true, y_pred, k=5)
# def pre_model():
#     # base_model = ResNet50(input_shape=(img_size, img_size, 3), weights=None, classes=5005)
#     base_model = InceptionV3(include_top=False,input_shape=(img_size, img_size, 3),weights=None)
#
#     base_model.add(Dense(32,input_shape=(,64)))
#     # base_model.compile(optimizer=Adam(lr=0.002), loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
#     return base_model
#
# model = pre_model()
# print(model.summary())

print(model.summary())
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

callback = [reduce_lr]
adam_z = optimizers.adam(lr=0.01)
model.compile(optimizer=adam_z, loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
history = model.fit(X, y, epochs=20, batch_size=1, verbose=1, validation_split=0.2, callbacks=callback)

# history = model.fit_generator(
#         train_generator,
#         steps_per_epoch=2000,
#         epochs=50,
#         validation_data=validation_generator,
#         validation_steps=800)


model.save('2w_balanced_model.h5')

plt.plot(history.history['top_5_accuracy'])
plt.plot(history.history['val_top_5_accuracy'])
plt.legend(['top_5_accuracy','val_top_5_accuracy'], loc='upper right')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('2w_balanced_1.jpg')


