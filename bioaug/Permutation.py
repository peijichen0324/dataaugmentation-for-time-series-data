import numpy as np


class Permutation(object):
    """Permutate the input time-series data randomly with a given probability.

    Args:
        p             (float) : Probability of applying permutation to the input signal.
        nPerm         (int, tuple, list)   : The number of segments into which the signal will be divided.
                                             The signal is split into 'nPerm' segments, which are then randomly reordered.
        minSegLength  (int, tuple, list)   : The minimum length of each segment.
    """

    def __init__(self, p=0.5, nPerm=4, minSegLength=10):
        self.p = p
        self.nPerm = nPerm
        self.minSegLength = minSegLength

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

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]"""
        if np.random.uniform(0, 1) < self.p:
            sequence_length, input_dim = signal.shape[0], signal.shape[1]
            signal_ = np.zeros((sequence_length, input_dim))
            nPerm = int(self._select_value(self.nPerm))
            minSegLength = int(self._select_value(self.minSegLength))

            idx = np.random.permutation(nPerm)
            flag = True
            while flag:
                segs = np.zeros(nPerm + 1, dtype=int)
                segs[1:-1] = np.sort(
                    np.random.randint(minSegLength, sequence_length - minSegLength, nPerm - 1))
                segs[-1] = sequence_length
                if np.min(segs[1:] - segs[:-1]) > minSegLength:
                    flag = False
            pp = 0
            for i in range(nPerm):
                signal_temp = signal[segs[idx[i]]:segs[idx[i] + 1], :]
                signal_[pp:pp + len(signal_temp), :] = signal_temp
                pp += len(signal_temp)
            return signal_
        return signal
