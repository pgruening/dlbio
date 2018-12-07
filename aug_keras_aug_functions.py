import keras.backend as K
from aug_augmentation_functions import IAugmentationFunction


class CropLabelToValidPaddingOutput(IAugmentationFunction):

    def setup(self, model):
        h_out, w_out = K.int_shape(model.cnn.layers[-1].output)[1:3]
        h_in, w_in = model.get_input_shape()[1:3]
        self.input_shape = (h_in, w_in)
        self.output_shape = (h_out, w_out)

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
        offset_h = (self.input_shape[0] - self.output_shape[0])//2
        offset_w = (self.input_shape[1] - self.output_shape[1])//2

        if offset_h == 0 and offset_w != 0:
            return label[offset_h:-offset_h, ...]

        elif offset_w == 0 and offset_h != 0:
            return label[:, offset_w:-offset_w, ...]

        elif offset_w == 0 and offset_h == 0:
            return label

        else:
            return label[offset_h:-offset_h, offset_w:-offset_w, ...]
