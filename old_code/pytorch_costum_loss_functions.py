from .pytorch_helpers import cuda_to_numpy
import numpy as np
from .costum_loss_functions import CostumMetric, INetworkLossFunction
import torch
import torch.nn.functional as F


class Accuracy(CostumMetric):

    def __init__(self):
        self.mode = 'max'
        self.__name__ = 'acc'

    def __call__(self, y_pred, y_true):
        y_pred = cuda_to_numpy(y_pred)
        y_pred = np.argmax(y_pred, axis=-1)
        y_true = cuda_to_numpy(y_true)
        y_true = y_true[..., 0]
        return np.mean(y_pred == y_true)


class DiceLoss(INetworkLossFunction):

    def __init__(self, class_weights=None):
        self.__name__ = 'dice_loss'
        self.mode = 'min'
        if class_weights is not None:
            self.class_weights = torch.tensor(
                class_weights,
                requires_grad=False).view(1,
                                          len(class_weights),
                                          1,
                                          1)
        else:
            self.class_weights = None

    def __call__(self, logits, true, eps=1e-7):
        """Computes the Sørensen–Dice loss.
        Note that PyTorch optimizers minimize a loss. In this
        case, we would like to maximize the dice loss so we
        return the negated dice loss.
        Args:
            true: a tensor of shape [B, 1, H, W].
            logits: a tensor of shape [B, C, H, W]. Corresponds to
                the raw output or logits of the model.
            eps: added to the denominator for numerical stability.
        Returns:
            dice_loss: the Sørensen–Dice loss.
        """
        num_classes = logits.shape[1]

        if num_classes == 1:
            true_1_hot = torch.eye(num_classes + 1)[true]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            true_1_hot_f = true_1_hot[:, 0:1, :, :]
            true_1_hot_s = true_1_hot[:, 1:2, :, :]
            true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)

            pos_prob = torch.sigmoid(logits)
            neg_prob = 1 - pos_prob

            probas = torch.cat([pos_prob, neg_prob], dim=1)
        else:

            true_1_hot = torch.eye(num_classes)[true]
            true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
            probas = F.softmax(logits, dim=1)

        dims = (0,) + tuple(range(2, true.ndimension()))
        probas = probas.cuda()

        # apply weights
        if self.class_weights is not None:
            probas = probas * self.class_weights

        true_1_hot = true_1_hot.cuda()
        intersection = torch.sum(probas * true_1_hot, dims)
        cardinality = torch.sum(probas + true_1_hot, dims)
        dice_loss = (2. * intersection / (cardinality + eps)).mean()
        return (1.0 - dice_loss.cuda())
