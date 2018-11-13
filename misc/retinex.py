import cv2
import numpy as np


def printInfo3(name, image):
    cv2.imwrite(name, image)
    min, max, bla, blub = cv2.minMaxLoc(np.ndarray.flatten(image))
    print(name, "min", min, "max", max)


def get_uint8_image(image):
    image = image.astype('float32')
    image -= np.min(image)
    image /= np.max(image)
    return (255*image).astype('uint8')


def MSRCR_scale(image, num_scales=7):
    height, width, num_channels = image.shape
    weight = 1./float(num_scales)
    output = np.zeros(image.shape)

    # image should be uint8 and is then set to range 1-256
    image = image.astype('float32') + 1
    for channel in range(num_channels):
        tmp = image[:, :, channel]
        for scale in range(num_scales):
            tmp = cv2.pyrDown(tmp)
            tmp_up = np.copy(tmp)
            for i in range(scale+1):
                tmp_up = cv2.pyrUp(tmp_up)
            tmp_up = cv2.resize(tmp_up, (width, height))
            output[:, :, channel] += weight * \
                (np.log(np.copy(image[:, :, channel]))-np.log(tmp_up))
        alpha = (128*image[:, :, channel]) / \
            (np.sum(image, axis=2)+float(num_channels))
        output[:, :, channel] *= np.log(alpha)
    return output


def scale_image(image,
                dynamic=1.2,
                use_fixed_scale=True,
                scale_val=.2):
    if use_fixed_scale:
        mini = -scale_val
        maxi = +scale_val
    else:
        mini = np.mean(image) - dynamic*np.std(image)
        maxi = np.mean(image) + dynamic*np.std(image)

    im_range = maxi - mini
    return ((255*(image - mini)/im_range).clip(min=0, max=255)).astype('uint8')


def image_retinex(im_orig,
                  use_fixed_scale=True,
                  scale_val=.2,
                  num_scales=7):
    """ Via a multiscale approach, computes a transformation which tries
        to achieve an equalized lighting of the image.
    """
    im_retinex = MSRCR_scale(im_orig, num_scales=num_scales)
    im_retinex = scale_image(im_retinex, use_fixed_scale)
    #printInfo3("retinex.png", im_retinex)
    return im_retinex


def retinex_unscaled(im_orig):
    return MSRCR_scale(im_orig)
