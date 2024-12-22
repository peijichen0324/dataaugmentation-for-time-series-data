import numpy as np
from scipy.interpolate import CubicSpline
import hashlib
import time


class TimeWarping(object):
    """Perform the TimeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of applying TimeWarping to the input signal.
        sigma    (float, tuple, list) : The standard deviation that controls the intensity of the time warping.
                                        A larger sigma value leads to more pronounced temporal distortions.
        knot     (int, tuple, list)   : The number of knots in the spline interpolation,
                                        determining the control points for the time warping.
    """

    def __init__(self, p=0.5, sigma=0.1, knot=4):
        self.p = p
        self.sigma = sigma
        self.knot = knot

    def _select_value(self, param):
        """Helper function to select a value from param if it's a tuple or list."""
        if isinstance(param, (int, float)):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        elif isinstance(param, list):
            return np.random.choice(param)
        else:
            raise ValueError("Parameter must be an int, float, tuple of length 2, or list.")
    
    @staticmethod
    def generate_seed():
        current_time = str(time.time() * 1000)
        hash_object = hashlib.md5(current_time.encode())
        hash_integer = int(hash_object.hexdigest(), 16)
        random_integer = hash_integer % 100
        return random_integer

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            seed = self.generate_seed()
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal.shape[0], signal.shape[1]

            sigma = self._select_value(self.sigma)
            knot = int(self._select_value(self.knot))
            tt_cum = self.init_seed(sequence_length, input_dim, seed, sigma, knot)
            if len(signal_.shape) == 1:
                signal_ = signal_[:, np.newaxis]

            signal_new = np.zeros((sequence_length, input_dim))
            x_range = np.arange(sequence_length)
            for i in range(input_dim):
                signal_new[:, i] = np.interp(x_range, tt_cum[:, 0], signal_[:, i])
            return signal_new
        return signal

    def init_seed(self, sequence_length, input_dim, seed, sigma, knot):
        x = (np.ones((input_dim, 1)) * (np.arange(0, sequence_length, (sequence_length - 1) / (knot + 1)))).transpose()
        np.random.seed(seed)
        y = np.random.normal(loc=1.0, scale=sigma, size=(knot + 2, input_dim))
        x_range = np.arange(sequence_length)
        tt = np.array([CubicSpline(x[:, i], y[:, i])(x_range) for i in range(input_dim)]).transpose()
        tt_cum = np.cumsum(tt, axis=0)
        # set the shape
        t_scale = [(sequence_length - 1) / tt_cum[-1, i] for i in range(input_dim)]
        for i in range(input_dim):
            tt_cum[:, i] = tt_cum[:, i] * t_scale[i]
        return tt_cum
