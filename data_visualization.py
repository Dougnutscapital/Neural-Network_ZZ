import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalAveragePooling2D
from keras.models import Model
import os
import matplotlib.pyplot as plt

df_train = pd.read_csv('train.csv')

print(df_train.head())

unique = pd.value_counts(df_train.Id)
print(unique.head())
num_classes = unique.values.shape[0]
print(unique.values)
print(num_classes)

# 去掉最多和最少
# load the data and add a column for occurrences
df = pd.read_csv('../train.csv')
grouped = df.groupby('Id')
print(grouped)
df['occurrences'] =  grouped.Id.transform('count')
print(df.shape)
# pick 10 samples from each class. One of these samples will go into the validation set.

# throw away all the labels which have fewer than 10 images associated with them and all the new whales
toy_df = df[(df.occurrences >= 10) & (df.Id != 'new_whale')]

# randomly pick 10 images for each label (without replacement)
random_pick_10 = lambda x: x.sample(10)

toy_df = toy_df.groupby('Id',group_keys=True).apply(random_pick_10)
print(toy_df)
print(toy_df.shape)
print(toy_df.index)

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

# X = prepareImages(toy_df, toy_df.shape[0], "../train")
X = prepareImages(toy_df, toy_df.shape[0], "../train")
# X /= 255
print(X.shape)

y, label_encoder = prepare_labels(toy_df['Id'])

print(y.shape)
print(y)



from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=90,
    featurewise_center=True,
    width_shift_range=0.2,
    height_shift_range=0.2)

train_datagen.fit(X)

from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense
from keras.metrics import categorical_accuracy, top_k_categorical_accuracy, categorical_crossentropy
from keras import optimizers

nb_classes = 273
FC_SIZE = 1024  # 全连接层的节点个数
NB_IV3_LAYERS_TO_FREEZE = 120  # 冻结层的数量

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

def top_5_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=5)

# 定义网络框架
base_model = InceptionV3(input_shape=(img_size, img_size, 3),weights='imagenet', include_top=False) # 预先要下载no_top模型
model = add_new_last_layer(base_model, nb_classes)              # 从基本no_top模型上添加新层
setup_to_transfer_learn(model, base_model)

print(model.summary())
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=False)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)

callback = [reduce_lr]
adam_z = optimizers.adam(lr=0.01,decay=0.001,momentum=0.9)
model.compile(optimizer=adam_z, loss='categorical_crossentropy', metrics=[categorical_crossentropy, categorical_accuracy, top_5_accuracy])
history = model.fit(X, y, epochs=10, batch_size=1, verbose=1, validation_split=0.2, callbacks=callback)

plt.plot(history.history['top_5_accuracy'])
plt.plot(history.history['val_top_5_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('1_toy.jpg')

model.save('toy_model_1.h5')



from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, \
    adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random


def randRange(a, b):
    '''
    a utility functio to generate random float values in desired range
    '''
    return pl.rand() * (b - a) + a


def randomAffine(im):
    '''
    wrapper of Affine transformation with random scale, rotation, shear and translation parameters
    '''
    tform = AffineTransform(scale=(randRange(0.75, 1.3), randRange(0.75, 1.3)),
                            rotation=randRange(-0.25, 0.25),
                            shear=randRange(-0.2, 0.2),
                            translation=(randRange(-im.shape[0] // 10, im.shape[0] // 10),
                                         randRange(-im.shape[1] // 10, im.shape[1] // 10)))
    return warp(im, tform.inverse, mode='reflect')


def randomPerspective(im):
    '''
    wrapper of Projective (or perspective) transform, from 4 random points selected from 4 corners of the image within a defined region.
    '''
    region = 1 / 4
    A = pl.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = pl.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))],
                  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1 - region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1 - region), im.shape[1])),
                   int(randRange(im.shape[0] * (1 - region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1 - region), im.shape[1])), int(randRange(0, im.shape[0] * region))],
                  ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)
    return warp(im, pt, output_shape=im.shape[:2])


def randomCrop(im):
    '''
    croping the image in the center from a random margin from the borders
    '''
    margin = 1 / 10
    start = [int(randRange(0, im.shape[0] * margin)),
             int(randRange(0, im.shape[1] * margin))]
    end = [int(randRange(im.shape[0] * (1 - margin), im.shape[0])),
           int(randRange(im.shape[1] * (1 - margin), im.shape[1]))]
    return im[start[0]:end[0], start[1]:end[1]]


def randomIntensity(im):
    '''
    rescales the intesity of the image to random interval of image intensity distribution
    '''
    return rescale_intensity(im,
                             in_range=tuple(pl.percentile(im, (randRange(0, 10), randRange(90, 100)))),
                             out_range=tuple(pl.percentile(im, (randRange(0, 10), randRange(90, 100)))))


def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))


def randomGaussian(im):
    '''
    Gaussian filter for bluring the image with random variance.
    '''
    return gaussian(im, sigma=randRange(0, 5))


def randomFilter(im):
    '''
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaler and applys it on the input image.
    filters include: equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    '''
    Filters = [equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian,
               randomIntensity]
    filt = random.choice(Filters)
    return filt(im)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    var = randRange(0.001, 0.01)
    return random_noise(im, var=var)


def augment(im, Steps=[randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]):
    '''
    image augmentation by doing a sereis of transfomations on the image.
    '''
    for step in Steps:
        im = step(im)
    return im


