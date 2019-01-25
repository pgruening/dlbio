""" 
Mainly decorated functions defined in augmentation_functions so that
they can be used easily with DataSequences.
Look for function descriptions in the respective files.
"""

# BUG ? Numpy needs to be imported otherwise keras will not run training
import numpy as np

from . import aug_augmentation_functions as aug_fcns
import warnings

# TODO: where to put wn_aug
#import wave_net_augmentation as wn_aug
# ------------decorators for use with DataSequence------------------------------


class IAugFunctionDecorator(object):
    """The FunctionDecorate expects an object that inherits the 
    IAugmentationFunction class. More specific the augmentation function
    should have a setup method to gather model-architecture information if 
    needed (look for example at the LabelCropping augmentation function).

    In most cases you need to implement the __init__ function, where you 
    specifiy 'func' which is the actual augmentation function. Afterwards,
    call this init function via super to take care of some details like setting
    the name.

    Raises
    ------
    NotImplementedError
        This is only an abstract interface. The actual __call__ function needs
        to be implemented

    """

    def __init__(self, func):
        self.func = func
        try:
            self.__name__ = self.func.__name__
        except Exception:
            self.__name__ = 'unknown_function'

    def setup(self, model):
        self.func.setup(model)

    def __call__(self, image, label):
        raise NotImplementedError

    def print_warning(self):
        warnings.warn(
            'If the input arguments do not match the __call__ or' +
            ' __init__ you most likely used the wrong augmetation Decorator class.')


class DecoratorImageOnly(IAugFunctionDecorator):

    def __call__(self, image, label):
        try:
            return self.func(image), label
        except Exception as identifier:
            print('Error at function: {}'.format(self.func.__name__))
            self.print_warning()
            raise Exception(identifier)


class DecoratorLabelOnly(IAugFunctionDecorator):
    def __call__(self, image, label):
        if label is None:
            return image, None
        else:
            try:
                return image, self.func(label)
            except Exception as identifier:
                print('Error at function: {}'.format(self.func.__name__))
                self.print_warning()
                raise Exception(identifier)


class DecoratorBothValues(IAugFunctionDecorator):
    def __call__(self, image, label):
        try:
            return self.func(image, label)
        except Exception as identifier:
            print('Error at function: {}'.format(self.func.__name__))
            self.print_warning()
            raise Exception(identifier)

# -----------augmentation functions---------------------------------------------


class Aug_LabelCropping(DecoratorLabelOnly):
    def __init__(self, input_h_w, output_h_w):
        func = aug_fcns.CropLabelToValidPaddingOutput(input_h_w, output_h_w)
        super(Aug_LabelCropping, self).__init__(func)


class Aug_Cropping(DecoratorBothValues):
    def __init__(self, dim_x, dim_y, do_zero_pad_smaller_images=False):
        func = aug_fcns.RandomCrop(dim_x, dim_y, do_zero_pad_smaller_images)
        super(Aug_Cropping, self).__init__(func)


class Aug_FullObjectCropping(DecoratorBothValues):
    def __init__(self, dim_x, dim_y):
        func = aug_fcns.DetectionCropOnlyFullObject(dim_x, dim_y)
        super(Aug_FullObjectCropping, self).__init__(func)


class Aug_RandomBrightness(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.BrightnessAugment()
        super(Aug_RandomBrightness, self).__init__(func)


class Aug_RandomContrast(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.RandomContrast()
        super(Aug_RandomContrast, self).__init__(func)


class Aug_AddNoise(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.AddNormalNoise()
        super(Aug_AddNoise, self).__init__(func)


class Aug_GetFirstItemOfLabelList(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.GetFirstItemOfLabelList()
        super(Aug_GetFirstItemOfLabelList, self).__init__(func)


class Aug_CCImageToBinaryLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.ConnectedComponentImageToBinaryImage()
        super(Aug_CCImageToBinaryLabel, self).__init__(func)


class Aug_CCImageToEdgeCellLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.CCImageToCellAndEdgeLabel()
        super(Aug_CCImageToEdgeCellLabel, self).__init__(func)


class Aug_CCImageToTransitionCellLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.CCImageToCellAndTransitionLabel()
        super(Aug_CCImageToTransitionCellLabel, self).__init__(func)


class Aug_CCImageToPairsLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.ComputePairwiseLabel()
        super(Aug_CCImageToPairsLabel, self).__init__(func)


class Aug_GetCCImageFromEICells(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.EIHepCellsToCCImage()
        super(Aug_GetCCImageFromEICells, self).__init__(func)


class Aug_FlipImages(DecoratorBothValues):
    def __init__(self, flip_lr=True, flip_ud=True):
        func = aug_fcns.RandomFlipImages(flip_lr, flip_ud)
        super(Aug_FlipImages, self).__init__(func)


class Aug_Rot90(DecoratorBothValues):
    def __init__(self):
        func = aug_fcns.RandomRot90()
        super(Aug_Rot90, self).__init__(func)


class Aug_DistLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.ComputePairwiseLabel()()
        super(Aug_DistLabel, self).__init__(func)


class Aug_GetFeatureMap(DecoratorLabelOnly):
    def __init__(self, index=0):
        func = aug_fcns.GetFeatureMap(index)
        super(Aug_GetFeatureMap, self).__init__(func)


class Aug_GetDetectSegLabel(DecoratorLabelOnly):
    def __init__(self, detection_shapes):
        def flat_binary(x):
            bin_image = aug_fcns.ConnectedComponentImageToBinaryImage()(x)
            bin_one_hot = aug_fcns.BinaryImageOneHotEncoding()(bin_image)
            return bin_one_hot.flatten()

        def func(x): return aug_fcns.ConcatFunctionOutputs()(
            x,
            [flat_binary,
             lambda z: aug_fcns.GetDetectionLabel(detection_shapes)(z)]
        )
        func = func
        super(Aug_GetDetectSegLabel, self).__init__(func)


class Aug_GetDetectionLabel(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.GetDetectionLabel()
        super(Aug_GetDetectionLabel, self).__init__(func)


class Aug_RandomClahe(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.RandomClahe()
        super(Aug_RandomClahe, self).__init__(func)


class Aug_RandomRetinex(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.RandomRetinex()
        super(Aug_RandomRetinex, self).__init__(func)


class Aug_RandomColorswap(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.RandomColorswap()
        super(Aug_RandomColorswap, self).__init__(func)


class Aug_Inverse(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.InverseImage()
        super(Aug_Inverse, self).__init__(func)


class Aug_SSDRescale(DecoratorBothValues):
    def __init__(self):
        func = aug_fcns.SSDRandomResizeInput()
        super(Aug_SSDRescale, self).__init__(func)


class Aug_Blur(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.RandomBlur()
        super(Aug_Blur, self).__init__(func)


class Aug_AddShadows(DecoratorImageOnly):
    def __init__(self):
        func = aug_fcns.AddRandomShadows()
        super(Aug_AddShadows, self).__init__(func)


class Aug_RelabelCCImageToSeries(DecoratorLabelOnly):
    def __init__(self):
        func = aug_fcns.RelabelCCImageToSeries()
        super(Aug_RelabelCCImageToSeries, self).__init__(func)


class Aug_DilateLabel(DecoratorLabelOnly):
    def __init__(self, k_size=3):
        func = aug_fcns.DilateLabel(k_size)
        super(Aug_DilateLabel, self).__init__(func)


"""
class Aug_WNFlip(DecoratorBothValues):
    def __init__(self):
        func = wn_aug.WNRandomFlipImages()
        super(Aug_WNFlip, self).__init__(func)


class Aug_WNRot90(DecoratorBothValues):
    def __init__(self):
        func = wn_aug.WNRandomRot90()
        super(Aug_WNRot90, self).__init__(func)
"""
