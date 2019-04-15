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
import time
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

NUM_BATCHES_UNTIL_PRINT = 50
SAVE_EPOCH = 1


class PyTorchTraining(ITraining):

    def __call__(self,
                 pytorch_model,
                 generator_train,
                 optimizer,
                 loss_functions,
                 lr_policy,
                 number_of_epochs,
                 save_path,
                 generator_val=None,
                 costum_metrics=None,
                 use_tensorboard=False,
                 training_callbacks=[],
                 model_checkpoint_validation_period=SAVE_EPOCH,
                 print_period_in_batches=NUM_BATCHES_UNTIL_PRINT
                 ):
        n = NUM_BATCHES_UNTIL_PRINT

        val_metrics = [] + loss_functions
        if costum_metrics is not None:
            val_metrics += costum_metrics

        for epoch in range(number_of_epochs):  # loop over the dataset multiple times

            if costum_metrics is not None:
                metric_values = dict()
                for metric in costum_metrics:
                    metric_values[metric.__name__] = .0
            else:
                metric_values = None

            running_loss = 0.0

            loss_values = dict()
            for loss_fcn in loss_functions:
                loss_values[loss_fcn.__name__] = .0

            start_time = time.time()
            for i, data in enumerate(generator_train):
                # get the inputs
                inputs, labels = data

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = pytorch_model(inputs.cuda())

                loss = torch.tensor(0.0).cuda()
                for loss_fcn in loss_functions:
                    value = loss_fcn(outputs, labels.cuda())
                    loss += value
                    loss_values[loss_fcn.__name__] += value.item()

                loss.backward()
                optimizer.step()

                if costum_metrics is not None:
                    for metric in costum_metrics:
                        val = metric(outputs, labels)
                        metric_values[metric.__name__] += val

                # print statistics
                running_loss += loss.item()
                if i % print_period_in_batches == print_period_in_batches-1:
                    if i > 0:
                        print_loss(epoch, i, running_loss,
                                   metric_values, loss_values)

            time_needed = time.time()-start_time
            print('Time needed: {}'.format(time_needed))
            print_loss(epoch, i, running_loss, metric_values, loss_values)

            lr_policy.step()

            if generator_val is not None:
                print('---')
                print('Validation')
                # checking metrics on the validation set
                tmp_metric_values = np.zeros(len(val_metrics))
                for i, data in enumerate(generator_val):
                    inputs, labels = data

                    # forward + backward + optimize
                    outputs = pytorch_model(inputs.cuda())
                    for j, metric_function in enumerate(val_metrics):
                        tmp = metric_function(outputs, labels)
                        tmp_metric_values[j] += tmp.item()

                for j in range(tmp_metric_values.shape[0]):
                    tmp_metric_values[j] /= float(i)
                    print('{}:{}'.format(
                        val_metrics[j].__name__, tmp_metric_values[j])
                    )
                print('---')

            # saving the model
            if epoch % model_checkpoint_validation_period == 0:
                torch.save(pytorch_model, save_path)
                print('saving model to {}'.format(save_path))

        # save at the end of training
        torch.save(pytorch_model, save_path)
        print('saving model to {}'.format(save_path))


def print_loss(epoch, i, running_loss, metric_values, loss_values):
    print('[%d, %5d] loss: %.6f' %
          (epoch + 1, i + 1, running_loss / i))

    for key, val in loss_values.items():
        print('[%d, %5d] %s: %.6f' %
              (epoch + 1, i + 1, key, val / i))

    if metric_values is None:
        return

    for key, val in metric_values.items():
        print('[%d, %5d] %s: %.6f' %
              (epoch + 1, i + 1, key, val / i))
