import numpy as np

DEBUG = False


class Fixed():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def schedule(self, epoch):
        return self.learning_rate


class Step():
    def __init__(self, learning_rate, decay, step):
        self.learning_rate = learning_rate
        self.decay = decay
        self.step = step

    def schedule(self, epoch):
        return self.learning_rate * np.power(
            self.decay, np.floor((1 + epoch) / self.step))


class Rotary():
    def __init__(self, max_val, min_val, frequency, n_epochs, decay=1.0):
        """ frequency: number of runs from 0 to 2pi."""
        self.a = .5*(max_val - min_val)
        self.b = .5*(max_val + min_val)
        self.frequency = float(frequency)
        self.n_epochs = float(n_epochs)
        self.decay = decay
        if DEBUG:
            print(self.a, self.b)

    def schedule(self, epoch):
        y = 2*np.pi * float(epoch)/self.n_epochs * self.frequency
        learning_rate = self.a*np.cos(y) + self.b
        self.a, self.b = self.a*self.decay, self.b*self.decay
        if DEBUG:
            print(epoch, learning_rate, .5*y/np.pi)
        return learning_rate
