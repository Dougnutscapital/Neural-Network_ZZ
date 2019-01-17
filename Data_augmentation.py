from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, \
    adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import pylab as pl
import pandas as pd
import numpy as np
import os
import cv2

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

df_train = pd.read_csv('train.csv')

print(df_train.tail(5))
# print(df_train.shape)
#
# unique = pd.value_counts(df_train.Id)
# print(unique.head())
# num_classes = unique.values.shape[0]
# print(unique.values)
# print(num_classes)

print(os.listdir('./train_enhance/'))
img_name = '0a00c7a0f.jpg'

im = pl.imread('./train_enhance/' + img_name)
len1 = len([equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity, randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop])

count = 0
for step in [equalize_adapthist, equalize_hist]:
    #, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity, randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]:
    im = step(im)
    name = str(count) + '_' + img_name
    # cv2.imwrite(name, im)
    pl.imshow(im)
    # pl.savefig('./train_enhance/' + name)
    # print(df_train[df_train['Image'] == img_name].Id)
    label = df_train[df_train['Image'] == img_name].Id.values
    print(label[0])

    print(name)
    print('/')
    print([name,label[0]])
    print(len([name,label[0]]))
    print('/')
    print(pd.DataFrame(np.array([name,label[0]]).reshape(1,2),columns=['Image','Id']))
    df_train = df_train.append(pd.DataFrame(np.array([name,label[0]]).reshape(1,2),columns=['Image','Id']),ignore_index=True)
    print(df_train.shape)
    count = count + 1

print(df_train.tail())



# F = pl.figure(figsize=(15,6))
# G = pl.GridSpec(2, 4, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.1, figure=F)
# ax = pl.subplot(G[0])
# ax.imshow(im)
# ax.set_axis_off()
# ax.set_aspect('equal')
# ax.set_title('original')
# for i, filt in enumerate([equalize_adapthist, equalize_hist, adjust_log, adjust_sigmoid, randomGamma, randomGaussian, randomIntensity]):
#     ax = pl.subplot(G[i+1])
#     ax.imshow(filt(im))
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     ax.set_title(filt.__name__ + ' on (original)')

# F = pl.figure(figsize=(15,9))
# G = pl.GridSpec(2, 3, left=.01, right=.99, bottom=0.05, top=0.9, wspace=.01, hspace=0.05, figure=F)
# ax = pl.subplot(G[0])
# ax.imshow(im)
# ax.set_axis_off()
# ax.set_aspect('equal')
# ax.set_title('original' + r'$\rightarrow$')
# for i, step in enumerate([randomAffine, randomPerspective, randomFilter, randomNoise, randomCrop]):
#     ax = pl.subplot(G[i+1])
#     im = step(im)
#     ax.imshow(im)
#     ax.set_axis_off()
#     ax.set_aspect('equal')
#     ax.set_title(step.__name__ + (r'$\rightarrow$' if i < 4 else ''))

# pl.savefig('2.jpg')