import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from .helpers import to_uint8_image
from .nn_pytorch_model import PytorchNeuralNetwork
from .pytorch_helpers import cuda_to_numpy


class CellSegmentationModel(PytorchNeuralNetwork):

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_input_shape(self):
        return self.input_shape

    def do_task(self, image, do_pre_proc):
        # patchwise prediction
        pred = self._predict(image, do_pre_proc)

        if self.post_process_fcn is not None:
            return self.post_process_fcn(pred)

        if self.num_classes > 2:
            return pred[..., 1:]
        else:
            return pred[..., 1]  # return only is_cell channel

    def get_num_classes(self):
        return self.num_classes

    def get_output_shape_for_patchwise_processing(self):
        return self.get_input_shape()

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

        net_out = self.cnn(input.unsqueeze(0))

        out_seg = net_out['seg']
        # hourglass
        if isinstance(out_seg, list):
            tmp = torch.zeros(out_seg[0].shape).float().cuda()
            # average maps
            for i in range(len(out_seg)):
                tmp += out_seg[i]
            out_seg = tmp / float(len(out_seg))

        out_seg = F.softmax(out_seg, dim=1)
        output = cuda_to_numpy(out_seg)

        if 'dir' in net_out.keys():
            out_dir = net_out['dir']
            # hourglass
            if isinstance(out_dir, list):
                out_dir = out_dir[-1]
            out_dir = cuda_to_numpy(out_dir)
            output = np.concatenate([output, out_dir], axis=-1)

        return output
