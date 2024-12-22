import numpy as np
from scipy.interpolate import CubicSpline
import hashlib
import time


class MagnitudeWarping(object):
    """Perform the MagnitudeWarping to the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : Probability of applying MagitudeWarping to the input signal.
        sigma    (float, tuple, list) : The standard deviation that controls the scale of the random curves.
                                        A larger sigma value leads to more significant variations in magnitude.
        knot     (int, tuple, list)   : The number of knots in the spline interpolation,
                                        determining the number of control points for generating the random curves.
                                        More knots result in more complex variations in the magnitude curves.
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

            sequence_length, input_dim = signal.shape[0], signal.shape[1]
            sigma = self._select_value(self.sigma)
            knot = int(self._select_value(self.knot))

            # Generate random curves based on the selected sigma and knot values
            self.x = (np.ones((input_dim, 1)) * (np.arange(0, sequence_length, (sequence_length-1)/(knot+1)))).transpose()
            np.random.seed(seed)
            self.y = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, input_dim))
            self.x_range = np.arange(sequence_length)
            self.randomCurves = np.array([CubicSpline(self.x[:, i], self.y[:, i])(self.x_range) for i in range(input_dim)]).transpose()

            signal_ = np.array(signal).copy()
            return signal_ * self.randomCurves
        return signal
