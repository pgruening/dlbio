import argparse
import copy
import glob
import json
import os
import warnings

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras import backend as K

from . import (aug_augmentation_functions, helpers, learning_rate_policy,
               process_image_patchwise)
from .helpers import safe_division
from .misc.keras_optimizer import NormalizedSGD


class KerasNeuralNetwork(object):

    def __init__(self,
                 model_id,
                 pre_process_function,
                 setup_function,
                 post_process_fcn=None
                 ):
        """ Basic Neural Network (nn_) for instance segmentation.
        Consists of a pre-processing function (pref_*),
        a post-processing function (postf_*) and
        a keras model that can be trained.

        Parameters
        ----------
        model_id : str
          Name of the model
        loss_function : function(y_pred, y_target)
          A loss function that returns a loss value for y_pred and y_target
        pre_process_function : function(image)
          A function that takes an image as input an returns a processed image
         (e.g. normalization)
        setup_function : function()
          Returns a keras model when called
        optimizer :

        lr_policy : 

        costum_metrics: tuple of (function, str)
          any costum metric function you want to apply to training and early
          stopping. Is passed as a tuple with the function in position 0 and
          the mode str which is either 'min' or 'max' depending on wether the
          function needs to be maximized or minimized. E.g. (costum_acc, 'max')
        training_callbacks :

        use_accuracy : 

        model_checkpoint_validation_period :

        tensorboard_batch_size : 

        """

        self.ID = model_id

        self.pre_process_function = pre_process_function
        print(self.pre_process_function.__repr__())

        self.initial_weights = []

        self.setup_function = setup_function

        self.cnn = None

        self.post_process_fcn = post_process_fcn

    def do_task(self, input, do_pre_proc):
        raise NotImplementedError

    def setup_cnn(self):
        # NOTE: don't forget to run _save_initial_weights in this function
        raise NotImplementedError

    def get_output_shape_for_patchwise_processing(self):
        # default_shape = self.cnn.layers[-1].output_shape
        raise NotImplementedError

    def get_num_classes(self):
        # default_num = self.cnn.layers[-1].output_shape[-1]
        raise NotImplementedError

    def _predict(self, input, do_pre_proc, predict_patch=False):
        """Compute the keras_model output for a batch or an image. 
        NOTE: this is the output without any post-processing,
        to get the full output you need to implement do_task

        Parameters
        ----------
        input : np.array of shape (batch, h, w, channel) or (h, w, channel)
          Image(s) to be processed by the network. If input is a batch of images,
          each image must match the specific network input shape.
        predict_patch : bool, optional
          if true, predict the input which needs to match the network input size.
          Otherwise, the image is processed patchwise (the default is False,
          which means an image of any size is returned with the same size).
        Returns
        -------
        np.array of shape (batch, h, w, out) or (h, w, out)
          output of the keras model
        """
        if do_pre_proc:
            print('Image is pre_processed')
            input = self.pre_process(input)
            if input.ndim == 2:
                input = input[..., np.newaxis]

        # single images can vary in size, hence might be processed patchwise
        is_single_image = input.ndim == 3
        if is_single_image:
            if predict_patch:
                return self._cnn_predict(input[np.newaxis, :, :, :])[0, :, :, :]
            else:
                return process_image_patchwise.whole_image_segmentation(
                    self,
                    input
                )
        else:
            return self._cnn_predict(input)

    def _cnn_predict(self, input):
        """Return the output of the keras model.
        May need a decorator for specific models.

        Parameters
        ----------
        input : np.array
            input for the network
        Returns
        -------
        np.array
            output of the keras model
        """
        return self.cnn.predict(input)

    def show_activations(self, image,
                         save_path="./activations",
                         max_fmaps_per_layer=64):

        def compute_activations(model, layer, input):
            get_activations = K.function([model.layers[0].input, K.learning_phase()],
                                         [layer.output]
                                         )
            activations = get_activations([input, 0])
            return activations[0]

        out_im = helpers.to_uint8_image(np.copy(image[0, ...]))
        cv2.imwrite(save_path+"/input_image.png",
                    out_im)

        activations = []
        for i, layer in enumerate(self.cnn.layers):
            if i == 0:
                continue

            is_uninteresting_layer = [x in layer.name for x in ['batch',
                                                                'concat',
                                                                'cropping',
                                                                'slicing',
                                                                'bn',
                                                                'up',
                                                                'pool',
                                                                'block',
                                                                'flatten',
                                                                'conf',
                                                                'loc']]
            if 'half' in layer.name:
                is_uninteresting_layer = [False]
            if True in is_uninteresting_layer:
                print('{} not shown.'.format(layer.name))
                continue

            layer_save_path = os.path.join(
                save_path, "{}_{}".format(str(i).zfill(3), layer.name))
            if not os.path.isdir(layer_save_path):
                os.makedirs(layer_save_path)
            print(layer_save_path)

            activations = compute_activations(self.cnn, layer, image)

            for n in range(min(activations.shape[-1], max_fmaps_per_layer)):
                plt.figure()
                plt.imshow(activations[0, ..., n])
                plt.colorbar()
                plt.savefig(layer_save_path + "/{}.png".format(n))
                plt.close()

    def reset_model(self):
        if not self.initial_weights:
            raise ValueError("No initial weights loaded.")

        for i, layer_weights in enumerate(self.initial_weights):
            self.cnn.layers[i].set_weights(layer_weights)

    def _save_initial_weights(self):
        self.initial_weights = []
        for layer in self.cnn.layers:
            self.initial_weights.append(layer.get_weights())

    def get_input_shape(self):
        return self.cnn.layers[0].output_shape

    def load(self, model_file):
        # there have been OOM issues when loading a model if
        # a model is already in memory
        if self.cnn is not None:
            self.cnn = None
            K.clear_session()

        self.cnn = keras.models.load_model(
            model_file
        )
        print('loaded model: {}'.format(model_file))
        self.cnn.summary()
        self._save_initial_weights()

    def pre_process(self, input):
        return self.pre_process_function(input)
