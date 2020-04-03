""" Augmentation functions that can be used in a Data Sequence.
    There is a distinction between label and input augmentations.
    To be used with a DataSequence, some of the functions need to be decorated.
    You find the decorated functions in aug_fcn_for_generator.py
"""
import warnings
from random import choice

import cv2
import numpy as np
import matplotlib.pyplot as plt

# TODO: where to put ei code
#from projects.ei_hep.ei_hep_make_data import CELL_IDX, NUC_IDX
from .misc.retinex import image_retinex
from .ssd.ssd_label_computations import (get_normalized_rectangles,
                                         get_ssd_labels_fast)
from .helpers import pRectangle, cell_labeling
import copy


class IAugmentationFunction(object):
    """Some the augmentation functions need information of the model including
    the keras_cnn (e.g. label_cropping). However, those information are not
    necessarily available when thei are called for the first time in get_dataset.
    To postpone the information gathering to a moment where the actual needed
    information are instantiated (right before being used in a DataSequence),
    each augmentation function is now a class that has a 'setup' method.
    This method is called before using the DataSequence. In most cases,
    this method does nothing.
    Furthermore, most of the methods are only triggered with a certain 
    probability. This behaviour is now encoded in a the call method. You need
    to specify the actual method in 'activated' and the return when there
    is no trigger in de_activated.

    Raises
    ------
    NotImplementedError
        Activated call is the actual function behaviour. 

    """

    def __init__(self, trigger_prob=.5, **kwargs):
        self.trigger_prob = trigger_prob
        self.__name__ = self.__class__.__name__

    def setup(self, model):
        pass

    def de_activated_call(self, *input):
        if len(input) == 1:
            return input[0]
        if len(input) == 2:
            return input[0], input[1]

    def activated_call(self, *input):
        raise NotImplementedError('Class: {}'.format(self.__class__.__name__))

    def __call__(self, *input):
        if np.random.rand() < self.trigger_prob:
            return self.activated_call(*input)
        else:
            return self.de_activated_call(*input)


############################# any input augmentation ###########################


class StackFunctionOutputs(IAugmentationFunction):
    def __call__(self, input, function_list):
        """Gather ouputs of a list of functions and stack them along
        the last axis to a tensor.

        Parameters
        ----------
        input : np.array
            input that is passed to each function
        function_list : list of functions
            list of functions that transform the input array. Note that
            each function's dimension must match, e.g. (b,h,w).

        Returns
        -------
        np.array
            adds a new axis to the tensor, e.g three function outputs with (b,h,w)
            are stacked to output (b,h,w,3)
        """
        outputs = []
        for func in function_list:
            outputs.append(func(input))

        return np.stack(outputs, axis=-1)


class ConcatFunctionOutputs(IAugmentationFunction):
    def __call__(self, input, function_list):
        """Gather output and concatenate them along the last axis.

        Parameters
        ----------
        input : np.array
            input that is passed to each function
        function_list : list of functions
            list of functions that transform the input array. Note that
            the first dimension of each function must match, e.g. (b,h,w, 4) and
            (b,h,w,10)

        Returns
        -------
        np.array
            Return a new tensor. With last dim = sum of each functions output.
        """
        outputs = []
        for func in function_list:
            outputs.append(func(input))

        return np.concatenate(outputs, axis=-1)


############################# Label augmentations ##############################

class DilateLabel(IAugmentationFunction):
    def __init__(self, k_size=3):
        self.kernel = np.ones((k_size, k_size))

    def __call__(self, label):
        return cv2.dilate(label, self.kernel)


class GetFirstItemOfLabelList(IAugmentationFunction):
    def __call__(self, label):
        if label is None:
            return None
        else:
            return label[0]


class CropLabelToValidPaddingOutput(IAugmentationFunction):

    def __init__(self, input_h_w, output_h_w):
        """Label cropping that can be used if input shape and output
        shape are given manually. There is a keras version in
        aug_keras_aug_functions.py


        Parameters
        ----------
        input_h_w : tuple:(int,int)
            height and width of the input
        output_h_w : tuple:(int,int)
            height and width of the output

        """
        self.input_h_w = input_h_w
        self.output_h_w = output_h_w

    def __call__(self, label):
        """Used when network returns a smaller output than input. Which is common,
        when e.g. valid padding is used.

        Parameters
        ----------
        label : np.array of input shape
            bigger label that needs to be fitted to the smaller output
        input_shape : np.array or list with [h, w, ...]
            shape of the network's input
        output_shape : np.array or list with [h_out, w_out, ...]
            shape of the network's output
        Returns
        -------
        np.array of size (h_out, w_out, dim)
            returns cropped label.
        """
        offset_h = (self.input_h_w[0] - self.output_h_w[0])//2
        offset_w = (self.input_h_w[1] - self.output_h_w[1])//2

        if offset_h == 0 and offset_w != 0:
            return label[offset_h:-offset_h, ...]

        elif offset_w == 0 and offset_h != 0:
            return label[:, offset_w:-offset_w, ...]

        elif offset_w == 0 and offset_h == 0:
            return label

        else:
            return label[offset_h:-offset_h, offset_w:-offset_w, ...]


class GetDetectionLabel(IAugmentationFunction):
    def __init__(self):
        self.priors = None
        self.__name__ = 'get_detection_label'

    def setup(self, model):
        self.priors = model.priors

    def __call__(self, label):
        """" Given a connected component image, return detection box labels
        for a single shot detector network.

        Parameters
        ----------
        label : cc_image
            connected_component image, each pixel value defines to which instance
            the pixel belongs
        priors : list of priors
            a prior is a tuple of (h, w) referring to the shape of 
            the respective output feature-map and a list of boxSpecs defining
            Anchorboxes. 

        Returns
        -------
        array of shape (n,)
            flattened array the contains confidence and offset feature maps
            that can be used for a SSD Network.
        """
        return get_ssd_labels_fast(label, self.priors)


class GetFeatureMap(IAugmentationFunction):
    def __init__(self, index, keep_dims=False):
        self.index = index
        self.keep_dims = keep_dims
        super(GetFeatureMap, self).__init__(trigger_prob=-1.0)

    def __call__(self, label):
        """Return only the feature map at position index

        Parameters
        ----------
        label : np.array of shape (...,h, w, dim)
            input array
        index : int
        Returns
        -------
        new label label : np.array of shape(...,h ,w)
            returns label with 
        """
        new_label = label[..., self.index]
        if self.keep_dims:
            new_label = new_label[..., np.newaxis]
        return new_label


class DownsampleLabel(IAugmentationFunction):
    def __call__(self, label, factor=2):
        """Downsample input by factor 

        Parameters
        ----------
        label : np.array of shape (h, w, dim)
            input array to be downsamplep along height and width.
        factor : int, optional
            only factor-th pixel is kept (the default is 2).

        Returns
        -------
        np.array of shape (h//factor, w//factor, dim)
            returns downsampled input
        """
        return label[::factor, ::factor, ...]


class ConnectedComponentImageToBinaryImage(IAugmentationFunction):
    def __call__(self, cc_image):
        """ 
        Given an image with background marked as zero and a specific object marked
        with an index > 0. Return a binary image with background equal to zero and 
        foreground equal to one.
        Parameters
        ----------
        cc_image : np.array of dtype uint8, int32, int64 or float32
            Zero is background. A specific object is marked with an integer index
            greater than zero.
        Returns
        -------
        binary_image: np.array float32
            Image describing background (0) and object (1)
        """
        return np.ceil(cc_image.clip(min=0, max=1).astype('float32'))


class BinaryImageOneHotEncoding(IAugmentationFunction):
    def __call__(self, binary_image):
        h, w = binary_image.shape[0], binary_image.shape[1]
        if binary_image.ndim == 3:
            binary_image = binary_image[..., 0]
        output = np.zeros((h, w, 2))
        output[..., 0] = 1 - binary_image
        output[..., 1] = binary_image
        return output


class ComputeNormalVectorLabel(IAugmentationFunction):
    def __call__(self, cc_image):
        """For each object pixel compute the normal vector that points to the 
        nearest boundary pixel, which is either a background pixel or a pixel
        belonging to another cell.

        Parameters
        ----------
        cc_image : np.array of shape (h, w, dim)
            Zero is background. A specific object is marked with an integer index
            greater than zero.
        Returns
        -------
        normal_vector_label : np.array of shape (h, w, 2) and dtype float32
            first feature map is the x-component, second is the y-component
            of the vector.
        """
        h, w = cc_image.shape
        normal_vector_image = np.zeros((h, w, 2))

        # for each cell pixel
        for y in range(h):
            for x in range(w):

                current_id = cc_image[y, x]

                if current_id == 0:
                    continue

                # find nearest pixel that belongs to another segment
                nearest_pixel_found = False

                found_pixel = [0, 0]
                min_distance = np.inf

                n = 1
                while not nearest_pixel_found:

                    for j in np.arange(-n, n):
                        if y + j < 0 or y + j >= h:
                            continue

                        # only look at outermost rectangle -> [+/-n, i] or [j, +/-n]
                        if abs(j) == n:
                            indeces = np.arange(-n, n)
                        else:
                            indeces = [-n, n]

                        for i in indeces:
                            if x + i < 0 or x + i >= w:
                                continue

                            pixel = cc_image[y+j, x+i]
                            if pixel != current_id:

                                current_distance = j**2 + i**2

                                if current_distance < min_distance:
                                    min_distance = current_distance
                                    found_pixel = [j, i]

                    n += 1
                    # nearest pixel = 0**2 + n**2; if greater than min_distance,
                    #  there is no sense in looking further
                    if n**2 >= min_distance:
                        nearest_pixel_found = True

                # compute normal vector
                vector_norm = np.sqrt(float(min_distance))
                for i in [0, 1]:
                    normal_vector_image[y, x, i] = float(
                        found_pixel[i])/vector_norm

        return normal_vector_image


class ComputePairwiseLabel(IAugmentationFunction):
    def __init__(self, k):
        self.k = k

    def __call__(self, cc_image):
        """Computes label needed for local distance approach. Values
        are in {-1, +1}. For each pixel a value comparison is done in the
        local neighborhood specified by k. Equal pixels are set to +1,
        else -1. 

        Parameters
        ----------
        cc_image : np.array of shape (h, w, dim)
            pixel_values define to which cell the pixel belongs
        k : int, optional
            size of the local neigborhood.
            A simple square around a pixel of size (k,k)
            (the default is config.get('local_dist_k')).

        Returns
        -------
        array of shape (h, w, k**2)
        each featuremap of the output denotes the if pixels of a certain
        location are equal. For example in k=3
        [0 1 2] the featuremap at 0, is the comparison of pixel 0 to 4
        [3 4 5] the featuremap at 7, is the comparison of pixel 0 to 7
        [6 7 8] etc.

        """

        h, w = cc_image.shape[0], cc_image.shape[1]
        output = np.zeros((h, w, self.k**2), dtype='float32')
        c = k//2
        if cc_image.ndim == 2:
            pad_label = np.lib.pad(cc_image, ((c, c+1), (c, c+1)), 'constant')
        else:
            pad_label = np.lib.pad(
                cc_image, ((c, c), (c, c), (0, 0)), 'constant')

        # TODO: distance weights nearer points -> more important
        weigths = np.ones(self.k**2)
        #weigths = _get_distance_weights(k)

        for y in range(h):
            for x in range(w):
                y_p, x_p = y+c, x+c

                point_in_normal_image = cc_image[y, x]
                patch_in_padded_image = pad_label[y_p-c:y_p+c+1, x_p-c:x_p+c+1]

                point_is_cell = point_in_normal_image > 0
                same_val = patch_in_padded_image == point_in_normal_image

                if point_is_cell:
                    cell_to_other = np.logical_and(
                        patch_in_padded_image > 0,
                        np.logical_not(same_val)
                    )
                    cell_to_cell = np.logical_and(
                        patch_in_padded_image > 0,
                        same_val
                    )
                    cell_to_bg = patch_in_padded_image == 0

                else:
                    cell_to_other = np.zeros(
                        patch_in_padded_image.shape, dtype='bool')
                    cell_to_cell = np.zeros(
                        patch_in_padded_image.shape, dtype='bool')
                    cell_to_bg = np.zeros(
                        patch_in_padded_image.shape, dtype='bool')

                if point_in_normal_image == 0:
                    bg_to_cell = np.logical_not(np.copy(same_val))
                    bg_to_bg = np.copy(same_val)
                else:
                    bg_to_cell = np.zeros(
                        patch_in_padded_image.shape, dtype='bool')
                    bg_to_bg = np.zeros(
                        patch_in_padded_image.shape, dtype='bool')

                patch = np.zeros(patch_in_padded_image.shape, dtype='uint8')
                patch[cell_to_cell] = CELL_TO_CELL
                patch[cell_to_other] = CELL_TO_OTHER
                patch[cell_to_bg] = CELL_TO_BG
                patch[bg_to_bg] = BG_TO_BG
                patch[bg_to_cell] = BG_TO_CELL

                output[y, x, :] = patch.flatten()*weigths
        # TODO: (k+1)-th output is unecessary
        output[output == 0] = 0

        return output.astype('uint8')


def _get_distance_weights(k):
    """Linear distance weights for distance points.
    The farthest points gets weight .5, the nearest 1.0.

    Parameters
    ----------
    k : int
        local window size.
    """
    c = k//2
    a = -.5/(2.0*c**2)
    b = 1.0

    ctr = 0
    weights = np.zeros(k**2)
    for y in np.arange(-c, c+1):
        for x in np.arange(-c, c+1):
            sq_dist = y**2 + x**2
            weights[ctr] = a*float(sq_dist) + b
            ctr += 1

    return weights

# class EIHepCellsToCCImage(IAugmentationFunction):
#    def __call__(self, label):
#        """Compute a connected-components image from a label image with
#        classes indeces of the ei project. See ei_help_make_data.py
#        for more information.
#        Parameters
#        ----------
#        label : np.array of shape (h, w) or (h, w, 1)
#            each pixel as an index indicating to which class it belongs.
#        Returns
#        -------
#        cc_image: np.array of same shape
#            pixel_values define to which cell the pixel belongs, zero is background
#        """
#        cells = (label == CELL_IDX)
#        nucleus = (label == NUC_IDX)
#        cell_label = (cells + nucleus).astype('uint8')
#        cc_image = cv2.connectedComponents(cell_label)[1]
#        return cc_image

################################ input augmentation ############################


def rescaling_decorator(func):
    """Some of the cv2 functions can only be applied to uint8 images.
    If the input is of some other type, it is casted to uint8 here, processed 
    and rescaled and casted afterwards. Use this function as a decorator.

    Parameters
    ----------
    func : function
        augmentation function that is applied

    Returns
    -------
    np.array of type input
        returns fcn(input_image)
    """

    def do_rescaling(image, *values):
        if image.dtype != 'uint8':
            needs_rescaling = True
            old_dtype = image.dtype
            image.astype('float32')

            offset = np.min(image)
            image -= offset

            scale = np.max(image)
            image /= scale
            image = (255.0*image).astype('uint8')
        else:
            needs_rescaling = False

        rgb = func(image, *values)

        if needs_rescaling:
            rgb = rgb.astype('float32')
            rgb /= 255.0
            rgb *= scale
            rgb += offset
            rgb = rgb.astype(old_dtype)

        return rgb

    return do_rescaling


def add_position_grid(image):
    pass


# @rescaling_decorator
class AddRandomShadows(IAugmentationFunction):
    #def de_activated_call(self, image): return image

    def activated_call(self, image, alpha=0.45, gauss_sigma=15):
        shadows = np.random.rand(*image.shape)
        for _ in range(np.random.randint(low=1, high=3)):
            shadows = cv2.GaussianBlur(shadows, (0, 0), gauss_sigma)
        shadows -= np.min(shadows)
        shadows /= np.max(shadows)

        if image.ndim > 2 and image.shape[-1] == 1:
            image = image[..., 0]
        #output = shadows
        output = alpha*image + (1.0-alpha)*image*shadows

        # if input shape is (x,x,1) output will be of shape (x, x, x)
        # NOTE: hacky bugfix
        if output.ndim > 2 and output.shape[-1] != 3:
            return output[..., 0]

        return output


class AddNormalNoise(IAugmentationFunction):
    def activated_call(self, image, factor=.01, offset=0.0, use_uniform=True):
        """Add normally distributed noise to image

        Parameters
        ----------
        image : np.array of type uint8 or float
            input image. Noise is added with offset + factor*normal
        factor : float, optional
        offset: float, optional
        Returns
        -------
        noisy image: np.array same shape same type
            if image is uint8 values are clipped to 0 and 255
        """
        old_type = image.dtype
        image = image.astype('float64')

        f_with_var = factor*np.std(image)
        if use_uniform:
            image += f_with_var*(np.random.uniform(size=image.shape)-.5)
        else:
            image += f_with_var*(np.random.normal(size=image.shape))

        if old_type == 'uint8':
            image.clip(min=0, max=255)

        return image.astype(old_type)


class BrightnessAugment(IAugmentationFunction):
    def activated_call(self, image, factor=0.1):
        """Randomly change input images brightness.
        Image if transformed into HSV color space and brightness value
        is multiplied with (factor + random.uniform).

        Parameters
        ----------
        image : np.array
            image needs to be in uint8 in order to transform the colorspace.
            If image is of type float it will be temporarily rescaled and
            transformed to uint8

        factor : float, optional
            default value is .5 which means the image's brightness can either
            be reduced by half or increased by half.

        Returns
        -------
        np.array
            brightness adjusted image
        """
        # taken from https://www.kaggle.com/chris471/basic-brightness-augmentation

        if image.dtype != 'uint8':
            needs_rescaling = True
            old_dtype = image.dtype
            image.astype('float32')

            offset = np.min(image)
            image -= offset

            scale = np.max(image)
            image /= scale
            image = (255.0*image).astype('uint8')
        else:
            needs_rescaling = False

        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # convert to hsv
        hsv = np.array(hsv, dtype=np.float64)

        # scale channel V uniformly
        hsv[:, :, 2] = hsv[:, :, 2] * (factor + np.random.uniform())
        hsv[:, :, 2][hsv[:, :, 2] > 255] = 255  # reset out of range values
        rgb = cv2.cvtColor(np.array(hsv, dtype=np.uint8), cv2.COLOR_HSV2RGB)

        if needs_rescaling:
            rgb = rgb.astype(old_dtype)
            rgb /= 255.0
            rgb *= scale
            rgb += offset

        return rgb


class RandomContrast(IAugmentationFunction):
    def activated_call(self, image, alpha=.5, beta=2.0):
        """ Image is transformed with
        new_image = (alpha+rand)*image + beta*rand

        Parameters
        ----------
        image : np.array
            input image
        alpha : float, optional
            Multiplication offset (the default is .5)
        beta : float, optional
            Addition offset (the default is 2)

        Returns
        -------
        np.array
            Transformed image. If uint 8 values are clipped to [0, 255]
        """
        old_type = image.dtype
        image = image.astype('float64')
        image = (alpha+np.random.uniform())*image + beta*np.random.normal()
        if old_type == 'uint8':
            image.clip(min=0, max=255)
        return image.astype(old_type)


@rescaling_decorator
class InverseImage(IAugmentationFunction):
    def activated_call(self, image):
        return 255 - image


@rescaling_decorator
class RandomClahe(IAugmentationFunction):
    def activated_call(self, rgb):
        # draw params
        grid_size = np.random.randint(low=8, high=32)
        clip_limit = 20.0 + 60.0*np.random.rand()
        # apply clahe
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        lab_planes = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        lab_planes[0] = clahe.apply(lab_planes[0])
        lab = cv2.merge(lab_planes)
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return rgb


@rescaling_decorator
class RandomRetinex(IAugmentationFunction):
    def activated_call(self, image):
        scale_val = .1 + .9*np.random.rand()
        num_scales = np.random.randint(low=1, high=7+1)
        return image_retinex(
            image,
            scale_val=scale_val,
            num_scales=num_scales
        )


class RandomColorswap(IAugmentationFunction):
    def activated_call(self, rgb):
        perm = np.random.permutation(np.arange(3))
        out = np.copy(rgb)
        for i in range(3):
            out[..., i] = rgb[..., perm[i]]
        return out


class RandomBlur(IAugmentationFunction):
    def activated_call(self, image):
        sigma = 1.0*np.random.rand()
        return cv2.GaussianBlur(image, (0, 0), sigma)


################################# both inputs ##################################
class SSDRandomResizeInput(IAugmentationFunction):
    def activated_call(self, image, label):
        rectangles = get_normalized_rectangles(label)
        rectangles = [x.to_rectangle(image.shape[0],
                                     image.shape[1])
                      for x in rectangles]

        min_val = float(min([min(x.w, x.h) for x in rectangles]))
        max_val = float(max([max(x.w, x.h) for x in rectangles]))

        smallest_factor = 20.0/min_val
        biggest_factor = 55.0/max_val

        a = smallest_factor
        b = biggest_factor - smallest_factor
        x_factor = a + b*np.random.rand()
        y_factor = x_factor

        @rescaling_decorator
        def resize_image(image):
            return cv2.resize(image, (0, 0), fx=x_factor, fy=y_factor,
                              interpolation=cv2.INTER_LINEAR)

        def resize_label(label):
            return cv2.resize(label, (0, 0), fx=x_factor, fy=y_factor,
                              interpolation=cv2.INTER_NEAREST)

        image = resize_image(image)
        label = resize_label(label)

        return image, label


class RandomFlipImages(IAugmentationFunction):
    def __init__(self, do_lr_flip=True, do_ud_flip=True, trigger_prob=.5):
        self.do_lr_flip = do_lr_flip
        self.do_ud_flip = do_ud_flip
        super(RandomFlipImages, self).__init__(trigger_prob=trigger_prob)

    def __call__(self, image, label):
        """Mirror images along height or widht axis, or both.
        A flip is done with 50 percent probabilty

        Parameters
        ----------
        image : np.array
            sample input image
        label : np.array
            sample input label
        do_lr_flip : bool, optional
            can the image be flipped along y axis (the default is True).
        do_ud_flip : bool, optional
            can the image be flipped along x axis (the default is True).

        Returns
        -------
        (np.array, np.array) or (np.array, None)
        flipped image and label image, label 

        """
        if self.do_lr_flip and np.random.rand() < self.trigger_prob:
            image = np.fliplr(image)
            if label is not None:
                label = np.fliplr(label)

        if self.do_ud_flip and np.random.rand() < self.trigger_prob:
            image = np.flipud(image)
            if label is not None:
                label = np.flipud(label)

        return image, label


class RandomRot90(IAugmentationFunction):
    def __call__(self, image, label):
        """Randomly rotate image and label by 90 degrees

        Parameters
        ----------
        image : np.array
            sample input image
        label : np.array
            sample input label
        Returns
        -------
        (np.array, np.array) or (np.array, None)
        rotated image and label image, label 
        """
        for _n in range(3):

            if np.random.rand() < self.trigger_prob:
                image = np.rot90(image)

                if label is not None:
                    label = np.rot90(label)

        return image, label


class RandomCrop(IAugmentationFunction):
    def __init__(self, dim_x, dim_y, do_zero_pad_smaller_images=False):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.do_zero_pad_smaller_images = do_zero_pad_smaller_images
        super(RandomCrop, self).__init__(trigger_prob=-1.0)

    def __call__(self, image, label):
        """Randomly crop image and corresponding label
        to size (dim_y, dim_x, ...)

        Parameters
        ----------
        image : np.array of shape (h, w ,...)
            sample input image
        label : np.array of shape (h, w ,...)
            sample input label
        dim_x : int
            output width
        dim_y : int
            output height
        Returns
        -------
        (np.array, np.array) or (np.array, None)
        cropped image and label image, label 
        """
        h, w = image.shape[0], image.shape[1]

        if h < self.dim_y or w < self.dim_x:
            if self.do_zero_pad_smaller_images:
                new_im = np.zeros((self.dim_y, self.dim_x, image.shape[-1]))
                new_im[:h, :w, :] = image
                image = new_im
                h, w = image.shape[0], image.shape[1]

            else:
                raise ValueError(
                    'Image is to small for cropping with shape: {}'.format(
                        image.shape)
                )

        x = np.random.randint(low=0, high=w-self.dim_x+1)
        y = np.random.randint(low=0, high=h-self.dim_y+1)

        if image.ndim == 2:
            image = image[..., np.newaxis]

        if label is not None:
            if label.ndim == 2:
                label = label[:, :, np.newaxis]
            image = image[y:y+self.dim_y, x:x+self.dim_x, :]
            label = label[y:y+self.dim_y, x:x+self.dim_x, :]
            return image, label

        else:
            return image[y:y+self.dim_y, x:x+self.dim_x, :]


class DetectionCropOnlyFullObject(IAugmentationFunction):
    def __init__(self, dim_x, dim_y, trigger_prob=.5):
        self.dim_x = dim_x
        self.dim_y = dim_y
        super(DetectionCropOnlyFullObject, self).__init__(
            trigger_prob=trigger_prob)

    def __call__(self, image, label):
        # returns cRectangles
        gt_rectangles = get_normalized_rectangles(label)

        if not gt_rectangles:
            return RandomCrop(self.dim_x, self.dim_y)(image, label)

        h, w = image.shape[0], image.shape[1]
        do_choose_rectangle = np.random.rand() < self.trigger_prob

        if do_choose_rectangle:
            rect = copy.copy(choice(gt_rectangles))
            rect = rect.to_rectangle(h, w)

            # the point far right and down is when the two top left points are
            # equal: (a, b) = (x, y)
            low_x = max(rect.x+rect.w - self.dim_x, 0)
            low_y = max(rect.y+rect.h - self.dim_y, 0)

            # the point on the far left and up is when the down right points are
            # equal: (a+dim_x, b+dim_y) = (x+w, y+h)
            # -> maximal value for a = x + w - dim_x
            high_x = min(rect.x, w-self.dim_x)
            high_y = min(rect.y, h-self.dim_y)

            x = np.random.randint(low=low_x, high=high_x+1)
            y = np.random.randint(low=low_y, high=high_y+1)

        else:
            x = np.random.randint(low=0, high=w-self.dim_x+1)
            y = np.random.randint(low=0, high=h-self.dim_y+1)

        if label is not None:
            if label.ndim == 2:
                label = label[:, :, np.newaxis]
            label = label[y:y+self.dim_y, x:x+self.dim_x, :]

        # """
            for old_rect in gt_rectangles:
                #old_rect = old_rect.to_p_rectangle(h, w)
                old_rect = old_rect.to_rectangle(h, w)
                # check if rectangle is within the cropped image
                cropped_image = pRectangle(
                    x=x,
                    y=y,
                    h=self.dim_y,
                    w=self.dim_x
                )
                rect_is_in_image = old_rect.estimate_jaccard_index(
                    cropped_image
                ) > 0.0
                if not rect_is_in_image:
                    continue

                # check if borders are changed by crop
                was_changed = False
                new_rx = old_rect.x
                new_ry = old_rect.y
                new_rh = old_rect.h
                new_rw = old_rect.w
                # update rectangle.x
                if old_rect.x < x:
                    was_changed = True
                    new_rx = x
                    # We only change x here and x+w must remain unchanged.
                    # We need to update w so that x+w == x_new+w_new
                    new_rw = old_rect.x + old_rect.w - new_rx

                # update rectangle.y
                if old_rect.y < y:
                    was_changed = True
                    new_ry = y
                    new_rh = old_rect.y + old_rect.h - new_ry

                # maybe x was already changed. We continue the computations with
                # new_rx and new_rw. Again, we need to adjust w. So that x + w = z
                # update rectangle.w
                if new_rx + new_rw > x + self.dim_x:
                    was_changed = True
                    new_rw = x + self.dim_x - new_rx

                # update rectangle.h
                if new_ry + new_rh > y + self.dim_y:
                    was_changed = True
                    new_rh = y + self.dim_y - new_ry

                if was_changed:
                    # create a new rectangle and check its jaccard_index
                    new_rect = pRectangle(
                        x=new_rx,
                        y=new_ry,
                        h=new_rh,
                        w=new_rw
                    )

                    ji = new_rect.estimate_jaccard_index(old_rect)

                    needs_to_be_deleted = ji < .8

                    if not needs_to_be_deleted:
                        continue

                    # in the cropped label, (0,0) begins at (x,y). We subtract the
                    # offset and set the label area of the new rect to zero
                    new_rect.x -= x
                    new_rect.y -= y

                    label[new_rect.y:new_rect.y+new_rect.h,
                          new_rect.x:new_rect.x+new_rect.w] = 0

                else:
                    continue

            return image[y:y+self.dim_y, x:x+self.dim_x, ...], label

        else:
            return image[y:y+self.dim_y, x:x+self.dim_x, ...]


class RelabelCCImageToSeries(IAugmentationFunction):
    def __call__(self, cc_image):
        return cell_labeling(cc_image)
