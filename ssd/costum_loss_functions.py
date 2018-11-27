import numpy as np
from keras import backend as K
import tensorflow as tf
import keras
from ..costum_loss_functions import CostumMetric, INetworkLossFunction


class SSDMetric(CostumMetric):
    def __init__(self, output_shapes):
        self.output_shapes = output_shapes
        self.__name__ = 'not_implemented'
        self.mode = 'max_or_min'
        raise NotImplementedError

    def confidence_evaluation_function(self, true_conf, pred_conf):
        raise NotImplementedError

    def __call__(self, y_true, y_pred):
        loss = K.variable(0.0)
        counter = K.variable(0.0)
        i = 0
        bs = K.shape(y_true)[0]

        # when model is loaded an error is thrown
        if self.output_shapes is None:
            return K.sum(y_pred)

        for s in self.output_shapes:
            num_entries = s[0]*s[1]*s[2]
            s = [bs, s[0], s[1], s[2]]

            tmp_pred = K.reshape(y_pred[:, i:i+num_entries], s)
            tmp_true = K.reshape(y_true[:, i:i+num_entries], s)
            i += num_entries

            num_boxes = int(s[-1]/6)
            for j in range(num_boxes):
                # get confidence maps
                tmp_true_conf = tmp_true[..., 2*j:2*j+2][..., 1]
                tmp_pred_conf = tmp_pred[..., 2*j:2*j+2][..., 1]

                loss += self.confidence_evaluation_function(
                    tmp_true_conf, tmp_pred_conf)
                counter += 1.0

        return loss/counter


class DetectionConfRecall(SSDMetric):
    def __init__(self, output_shapes):
        self.output_shapes = output_shapes
        self.__name__ = 'detection_conf_recall'
        self.mode = 'max'

    def confidence_evaluation_function(self, true_conf, pred_conf):
        return recall(true_conf, pred_conf)


def recall(true_conf, pred_conf):
    true_conf = K.cast(
        K.greater_equal(true_conf, 0.5), 'float32')

    TPs = K.sum(pred_conf * true_conf)
    TPs_FNs = K.sum(true_conf)
    return TPs / (TPs_FNs + .00001)


class DetectionConfPrecision(SSDMetric):
    def __init__(self, output_shapes):
        self.output_shapes = output_shapes
        self.__name__ = 'detection_conf_precision'
        self.mode = 'max'

    def confidence_evaluation_function(self, true_conf, pred_conf):
        return precision(true_conf, pred_conf)


def precision(true_conf, pred_conf):
    true_conf = K.cast(K.greater_equal(true_conf, 0.5), 'float32')

    TPs = K.sum(pred_conf * true_conf)
    TPs_FPs = K.sum(pred_conf)

    return TPs / (TPs_FPs + .00001)


class DetectionConfAccuracy(SSDMetric):
    def __init__(self, output_shapes):
        self.output_shapes = output_shapes
        self.__name__ = 'detection_conf_acc'
        self.mode = 'max'

    def confidence_evaluation_function(self, true_conf, pred_conf):
        return f1_score(true_conf, pred_conf)


def f1_score(true_conf, pred_conf):
    prec = precision(true_conf, pred_conf)
    rec = recall(true_conf, pred_conf)
    return 2.0*prec*rec/(prec + rec + .00001)


class DetectionLoss(INetworkLossFunction):
    def __init__(
            self,
            output_shapes,
            confidence_loss_weight=.5,
            localization_loss_weight=.5,
            positive_to_negative_ratio=.33,
            positives_weight=1.0
    ):
        self.__name__ = 'detection_loss'
        self.output_shapes = output_shapes

        self.conf_loss = CategoricalCrossentropy()
        self.l1_loss = SmoothL1()

        self.w_cl = confidence_loss_weight
        self.alpha = localization_loss_weight
        self.p_n_ratio = positive_to_negative_ratio

        self.pos_weight = positives_weight

    def __call__(self, y_true, y_pred):
        loss = K.variable(0.0)
        i = 0
        bs = K.shape(y_true)[0]

        # when model is loaded an error is thrown
        if self.output_shapes is None:
            return K.sum(y_pred)

        num_box_shapes = float(len(self.output_shapes))
        for s in self.output_shapes:
            num_entries = s[0]*s[1]*s[2]
            s = [bs, s[0], s[1], s[2]]

            tmp_pred = K.reshape(y_pred[:, i:i+num_entries], s)
            tmp_true = K.reshape(y_true[:, i:i+num_entries], s)
            i += num_entries

            num_boxes = int(s[-1]/6)
            o = 2*num_boxes
            for j in range(num_boxes):
                # get confidence maps
                tmp_true_conf = tmp_true[..., 2*j:2*j+2]
                tmp_pred_conf = tmp_pred[..., 2*j:2*j+2]

                # first feature_map is negative class
                is_match = tmp_true_conf[..., 1][..., np.newaxis]
                num_matches = K.sum(is_match)

                # hard negative mining: take only the N samples with the
                # highest error and num_matches/num_negatives = self.p_n_ratio
                is_neg = tmp_true_conf[..., 0][..., np.newaxis]
                num_negatives = K.sum(is_neg)
                scaled_num_negatives = K.round(num_matches/self.p_n_ratio)
                scaled_num_negatives = tf.minimum(
                    num_negatives, scaled_num_negatives)

                # if there are no positives examples in the image, just train
                # all negatives.
                num_negatives = tf.cond(
                    tf.less_equal(num_matches, 1),
                    lambda: tf.identity(num_negatives - 1.0),
                    lambda: tf.identity(scaled_num_negatives)
                )

                neg_pred = tmp_pred_conf[..., 0][..., np.newaxis]
                only_negatives_pred = is_neg*neg_pred

                # To exclude positive values:
                # positives values are set to 2. That means they will
                # appear as last values in top_k(-flat)
                only_negatives_pred_flat = K.flatten(
                    only_negatives_pred + 2.0*is_match)

                # NOTE: old bug?
                # only_negatives_pred_flat = K.flatten(
                #    only_negatives_pred + is_neg)

                # sort by minimum
                values, _ = tf.nn.top_k(
                    -1.0*only_negatives_pred_flat, K.cast(num_negatives, 'int32'))

                # values is a sorted vector of length num_negatives
                threshold = -1.0*values[-1]
                is_hard_negative = K.cast(
                    K.less_equal(only_negatives_pred, threshold),
                    'float32')
                # make sure its only negatives
                is_hard_negative = is_hard_negative*is_neg

                # use all positives and the n worst negatives
                used_values = (is_match + is_hard_negative)
                tmp_true_conf_weighted = tmp_true_conf * used_values

                # you can re-weight the positives samples
                weight = np.ones(K.int_shape(tmp_true_conf_weighted)[1:])
                weight[..., 1] *= self.pos_weight
                tmp_true_conf_weighted *= weight

                conf_loss = K.sum(self.conf_loss(
                    tmp_true_conf_weighted, tmp_pred_conf))
                conf_loss /= K.sum(used_values)

                loss += self.w_cl * conf_loss / \
                    float(num_boxes) / num_box_shapes

                tmp_true_loc = tmp_true[..., o+4*j:o+4*(j+1)]
                tmp_pred_loc = tmp_pred[..., o+4*j:o+4*(j+1)]

                l1 = self.l1_loss(tmp_true_loc, tmp_pred_loc, is_match)

                # add offset in case num_matches is zero
                l1 /= (num_matches + 1e-9)
                loss += self.alpha * l1 / float(num_boxes) / num_box_shapes

        return loss

#------------------------------------------------------------------------------#
#-------------------------No detection loss------------------------------------#
#------------------------------------------------------------------------------#


class DetectSegLoss(INetworkLossFunction):

    def __init__(
        self,
        number_of_segmentation_classes,
        output_shapes=None,
        **kwargs
    ):

        # needs to be loaded like this for config changes to work
        seg_loss_weight = kwargs.pop('seg_loss_weight', config.get('seg_loss'))
        conf_loss_weight = kwargs.pop(
            'conf_loss_weight', config.get('conf_loss'))
        alpha = kwargs.pop('alpha', config.get('alpha'))
        pos_neg_ratio = kwargs.pop(
            'pos_neg_ratio', config.get('pos_neg_ratio'))
        pos_weight = kwargs.pop('pos_weight', config.get('pos_weight'))

        self.output_shapes = output_shapes

        self.seg_loss = CategoricalCrossentropy()
        self.conf_loss = CategoricalCrossentropy()

        self.l1_loss = SmoothL1()

        self.w_sl = seg_loss_weight
        self.w_cl = conf_loss_weight
        self.alpha = alpha

        self.__name__ = 'detect_seg_loss'

        self.p_n_ratio = pos_neg_ratio
        self.pos_weight = pos_weight

    def __call__(self, y_true, y_pred):
        is_segmentation = True
        loss = K.variable(0.0)
        i = 0
        bs = K.shape(y_true)[0]
        # bs = K.int_shape(y_true)[0]
        # bs = 4

        # when model is loaded an error is thrown
        if self.output_shapes is None:
            return -1.0 + 0.0*K.sum(y_pred)

        num_box_shapes = float(len(self.output_shapes) - 1)
        for s in self.output_shapes:
            num_entries = s[0]*s[1]*s[2]
            s = [bs, s[0], s[1], s[2]]

            tmp_pred = K.reshape(y_pred[:, i:i+num_entries], s)
            tmp_true = K.reshape(y_true[:, i:i+num_entries], s)
            i += num_entries

            # first output shape is segmentation
            if is_segmentation:
                seg_loss_val = K.mean(self.seg_loss(tmp_true, tmp_pred))
                loss += self.w_sl*seg_loss_val
                is_segmentation = False
                continue

            num_boxes = s[-1]/6
            o = 2*num_boxes
            for j in range(num_boxes):
                # get confidence maps
                tmp_true_conf = tmp_true[..., 2*j:2*j+2]
                tmp_pred_conf = tmp_pred[..., 2*j:2*j+2]

                # first feature_map is negative class
                is_match = tmp_true_conf[..., 1][..., np.newaxis]
                num_matches = K.sum(is_match)

                """
                num_matches = tf.Print(
                    num_matches, [
                        s,
                        2*j,
                        num_matches,
                        num_boxes,
                    ],
                    summarize=100,
                    message='\n')
                """
                # hard negative mining: take only the N samples with the
                # highest error and num_matches/num_negatives = self.p_n_ratio
                is_neg = tmp_true_conf[..., 0][..., np.newaxis]
                num_negatives = K.sum(is_neg)
                scaled_num_negatives = K.round(num_matches/self.p_n_ratio)
                scaled_num_negatives = tf.minimum(
                    num_negatives, scaled_num_negatives)

                num_negatives = tf.cond(
                    tf.less_equal(num_matches, 1),
                    lambda: tf.identity(num_negatives),
                    lambda: tf.identity(scaled_num_negatives)
                )

                neg_pred = tmp_pred_conf[..., 0][..., np.newaxis]
                only_negatives_pred = is_neg*neg_pred
                # non negative values are set to 1
                only_negatives_pred_flat = K.flatten(
                    only_negatives_pred + 2.0*is_match)
                # sort by minimum
                values, _ = tf.nn.top_k(
                    -1.0*only_negatives_pred_flat, K.cast(num_negatives, 'int32'))

                # values is a sorted vector of length num_negatives
                threshold = -1.0*values[-1]
                is_hard_negative = K.cast(
                    K.less_equal(only_negatives_pred, threshold),
                    'float32')
                is_hard_negative = is_hard_negative*is_neg

                used_values = (is_match + is_hard_negative)

                tmp_true_conf_weighted = tmp_true_conf * used_values

                weight = np.ones(K.int_shape(tmp_true_conf_weighted)[1:])
                weight[..., 1] *= self.pos_weight
                tmp_true_conf_weighted *= weight

                # tmp_true_conf_weighted = tf.Print(
                #    tmp_true_conf_weighted,
                #    [K.max(tmp_true_conf_weighted)])

                conf_loss = K.sum(self.conf_loss(
                    tmp_true_conf_weighted, tmp_pred_conf))
                conf_loss /= K.sum(used_values)

                loss += self.w_cl * conf_loss / \
                    float(num_boxes) / num_box_shapes

                tmp_true_loc = tmp_true[..., o+4*j:o+4*(j+1)]
                tmp_pred_loc = tmp_pred[..., o+4*j:o+4*(j+1)]

                # l1 = K.sum(K.square(tmp_true_loc - tmp_pred_loc)*is_match)
                l1 = self.l1_loss(tmp_true_loc, tmp_pred_loc, is_match)
                # add offset in case num_matches is zero
                l1 /= (num_matches + 1e-9)
                loss += self.alpha * l1 / float(num_boxes) / num_box_shapes

        return loss


class SmoothL1(INetworkLossFunction):

    def __call__(self, y_true, y_pred, is_match=1.0):
        output = is_match*K.abs(y_true - y_pred)

        do_square = K.cast(K.less(output, 1.0), 'float32')
        do_square *= is_match

        do_abs = 1.0 - do_square
        do_abs *= is_match

        output = do_square*.5*output**2.0 + do_abs*(output - .5)
        return K.sum(output)


class WeightedSparseCategoricalCrossentropy(INetworkLossFunction):

    def __init__(self,
                 number_of_classes,
                 class_weights=None,
                 start_index_true=0,
                 stop_index_true=-1,
                 start_index_pred=0,
                 stop_index_pred=-1
                 ):
        self.number_of_classes = number_of_classes

        # class weights look like: [w0, w1, ...]
        if class_weights is None:
            class_weights = list(np.ones(number_of_classes))
        self.class_weights = class_weights

        self.start_true = start_index_true
        self.stop_true = stop_index_true
        self.start_pred = start_index_pred
        self.stop_pred = stop_index_pred

    def __call__(self, y_true, y_pred):
        def check_if_is_init():
            for s in shape:
                if s is None:
                    return True
            if y_true.shape[0] == 0:
                return True
            return False

        num_true_fmaps = K.int_shape(y_true)[-1]
        if num_true_fmaps > 1:
            y_true = y_true[..., self.start_true:self.stop_true]
        else:
            y_true = y_true[..., 0]
            y_true = K.cast(y_true, 'int32')

        y_pred = y_pred[..., self.start_pred:self.stop_pred]
        shape = K.int_shape(y_pred)

        is_init = check_if_is_init()

        if not is_init:
            y_true_one_hot = np.zeros(shape, dtype=np.float32)
            num_classes = y_pred.shape[-1]
            for i in range(num_classes):
                y_true_one_hot[y_true == i, i] = float(self.class_weights[i])

            return keras.losses.categorical_crossentropy(y_true_one_hot, y_pred)
        else:
            return K.sum(y_true)


class CategoricalCrossentropy(INetworkLossFunction):

    def __call__(self, y_true, y_pred):
        # turn vector to sparse integer representation
        return K.categorical_crossentropy(y_true, y_pred, from_logits=False)


class SquaredAngularLoss(INetworkLossFunction):
    def __init__(self,
                 start_index_true=0,
                 stop_index_true=2,
                 start_index_pred=0,
                 stop_index_pred=2,
                 binary_index=3):

        self.start_true = start_index_true
        self.stop_true = stop_index_true
        self.start_pred = start_index_pred
        self.stop_pred = stop_index_pred
        self.binary_index = binary_index

    def __call__(self, y_true, y_pred):

        y_pred = y_pred[..., self.start_pred:self.stop_pred]

        binary = y_true[..., self.binary_index]
        y_true = y_true[..., self.start_true:self.stop_true]

        # label is on position 0
        u_gt_1 = y_true[..., 0]
        u_gt_2 = y_true[..., 1]

        u_pred_1 = y_pred[..., 0]
        u_pred_2 = y_pred[..., 1]

        # normalize output vector
        norm = K.sqrt(u_pred_1*u_pred_1 + u_pred_2*u_pred_2 + 1e-9)

        u_pred_1 = u_pred_1/(norm)
        u_pred_2 = u_pred_2/(norm)

        scalar_product = u_gt_1*u_pred_1 + u_gt_2*u_pred_2

        # acos returns nan if input sufficiently near to -1 or 1
        scalar_product = tf.clip_by_value(scalar_product, -0.95, +0.95)

        # if sp=1 vectors are parallel, 0 -> orthogonal, -1 antiparallel
        # acos(1) = 0; acos(-1) = pi;
        angles = tf.acos(scalar_product)

        # is_dir_vector = tf.greater(tf.abs(u_gt_1) + tf.abs(u_gt_2), 0)
        is_dir_vector = tf.cast(binary, tf.float32)
        """
        is_dir_vector = tf.Print(is_dir_vector, [
            K.max(u_gt_1),
            K.max(u_gt_2),
            self.binary_index
        ]
        )
        """
        squared_angles = angles*angles*is_dir_vector
        # squared_angles = angles*is_dir_vector

        num_dir_vectors = tf.reduce_sum(is_dir_vector)
        mean_squared_angular_loss = tf.reduce_sum(
            squared_angles, name="mean_squared_angular_loss")/num_dir_vectors
        return mean_squared_angular_loss


class AccuracyOfIndex(CostumMetric):

    def __init__(self,
                 index_true=0,
                 start_index_pred=0,
                 stop_index_pred=1,
                 name='indexed_acc'):
        # set mode
        super(AccuracyOfIndex, self).__init__('max')

        self.index_true = index_true
        self.start_pred = start_index_pred
        self.stop_pred = stop_index_pred

        self.__name__ = name

    def __call__(self, y_true, y_pred):
        y_pred = y_pred[..., self.start_pred:self.stop_pred]
        y_pred = K.argmax(y_pred, axis=-1)
        y_pred = K.cast(y_pred, 'float32')

        y_true = y_true[..., self.index_true]

        return K.mean(tf.equal(y_true, y_pred))
