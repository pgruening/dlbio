# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from .uc_Itraining import ITraining
"""
Training a Classifier
=====================
1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Convolution Neural Network
3. Define a loss function
4. Train the network on the training data
5. Test the network on the test data

1. Loading and normalizing CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Using ``torchvision``, it’s extremely easy to load CIFAR10.
"""
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

NUM_BATCHES_UNTIL_PRINT = 10


class PyTorchTraining(ITraining):

    def __call__(self,
                 pytorch_model,
                 generator_train,
                 optimizer,
                 loss_function,
                 lr_policy,
                 number_of_epochs,
                 save_path,
                 generator_val=None,
                 costum_metrics=None,
                 use_tensorboard=False,
                 training_callbacks=[],
                 model_checkpoint_validation_period=1,
                 class_weight=None
                 ):
        n = NUM_BATCHES_UNTIL_PRINT

        val_metrics = [loss_function]
        if costum_metrics is not None:
            val_metrics += costum_metrics

        for epoch in range(number_of_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, data in enumerate(generator_train):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = pytorch_model(inputs.cuda())
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % n == n-1:    # print every n mini-batches
                    print('[%d, %5d] loss: %.3f' %
                          (epoch + 1, i + 1, running_loss / n))
                    running_loss = 0.0

            if generator_val is not None:
                # checking metrics on the validation set
                tmp_metric_values = np.zeros(len(val_metrics))
                for i, data in generator_val:
                    inputs, labels = data

                    # forward + backward + optimize
                    outputs = pytorch_model(inputs)
                    for j, metric_function in enumerate(val_metrics):
                        tmp = metric_function(outputs, labels)
                        tmp_metric_values[j] += tmp.item()

                for j in range(tmp_metric_values.shape[0]):
                    tmp_metric_values[j] /= float(i)
                    print('{}:{}'.format(
                        val_metrics[j].__name__, tmp_metric_values[j])
                    )

            # saving the model
            if epoch % model_checkpoint_validation_period == 0:
                torch.save(pytorch_model, save_path)
                print('saving model to {}'.format(save_path))
