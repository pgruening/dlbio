import argparse
import copy
import glob
import json
import os
import warnings

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

import torch
import torch.nn.functional as F
from DLBio.pytorch_helpers import cuda_to_numpy

from . import helpers, process_image_patchwise


class PytorchNeuralNetwork(object):

    def __init__(self,
                 model_id,
                 pre_process_function,
                 setup_function=None,
                 post_process_fcn=None,
                 to_tensor_fcn=transforms.ToTensor(),
                 normalization=None
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

        self.to_tensor = to_tensor_fcn

        self.num_classes = 2  # default for binary
        self.normalization = normalization

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
        input = self.to_tensor(input[0, ...]).float().cuda()
        if self.normalization is not None:
            input = self.normalization(input)

        output = self.cnn(input.unsqueeze(0))
        output = F.softmax(output, dim=1)
        output = cuda_to_numpy(output)
        return output

    def show_activations(self, image,
                         save_path="./activations",
                         max_fmaps_per_layer=64):
        raise NotImplementedError

    def reset_model(self):
        raise NotImplementedError

    def _save_initial_weights(self):
        raise NotImplementedError

    def get_input_shape(self):
        raise NotImplementedError

    def load(self, model_file, run_in_eval_mode=True):
        self.cnn = torch.load(
            model_file
        )
        if run_in_eval_mode:
            self.cnn.eval()

    def pre_process(self, input):
        return self.pre_process_function(input)
