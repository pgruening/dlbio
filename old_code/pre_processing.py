import cv2
import keras.applications.densenet as densenet
import keras.applications.mobilenet as mobilenet
import keras.applications.vgg16 as vgg16
import keras.applications.resnet50 as resnet50
import numpy as np

import DLBio.misc.retinex


def to_float_input(input):
    return input.astype('float32')


def no_pre_processing(input):
    return input


def pref_densenet(input):
    return densenet.preprocess_input(input.astype('float64'))


def pref_vgg16(input):
    return vgg16.preprocess_input(input.astype('float64'))


def pref_mobilenet(input):
    return mobilenet.preprocess_input(input.astype('float64'))


def pref_resnet50(input):
    return resnet50.preprocess_input(input.astype('float64'))


def pref_retinex(input):
    return retinex.image_retinex(input)


def pref_retinex_unscaled(input):
    return retinex.retinex_unscaled(input)


def pref_downsample(input, factor):

    if type(factor) != int:
        raise ValueError(
            'factor: {} needs to be an integer but is an {}'.format(
                factor, type(factor)
            )
        )

    else:
        return input[::factor, ::factor, ...]


def pref_to_grayscale(input):
    return cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[..., np.newaxis]


def pref_highpass_filter(input):
    input = input.astype('float32')
    low = cv2.GaussianBlur(input, (0, 0), 3)
    if low.ndim < input.ndim:
        low = low[..., np.newaxis]
    return input - low
