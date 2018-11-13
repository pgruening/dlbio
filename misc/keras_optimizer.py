from keras.legacy import interfaces
from keras.optimizers import Optimizer
import keras.backend as K


class NormalizedSGD(Optimizer):
    """Normalized Stochastic gradient descent optimizer.
    Includes support for momentum,
    learning rate decay, and Nesterov momentum.
    Supports two normalization modes: 'max' and 'l2'.
    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
        norm: 'max' or 'l2'. The way the gradient should be normalized.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False, norm='max', **kwargs):
        super(NormalizedSGD, self).__init__(**kwargs)

        if not norm in ['max', 'l2']:
            raise ValueError('Unexpected norm type %s.' % norm)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.norm = norm

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            # Gradient normalization
            if self.norm == 'max':
                g_max = K.max(K.abs(g), axis=None, keepdims=True)
                denominator = K.epsilon() + g_max
                g_step_normed = g / denominator
            else:
                g_step_normed = K.l2_normalize(g)

            v = self.momentum * m - lr * g_step_normed  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - lr * g_step_normed
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov,
                  'norm': self.norm
                  }
        base_config = super(NormalizedSGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

