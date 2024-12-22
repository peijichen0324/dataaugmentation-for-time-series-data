import numpy as np
import hashlib
import time


class SignalDrift(object):
    """Simulate signal drift by applying a linear or low-frequency drift to the input time-series data.

    Args:
        p         (float) : Probability of applying signal drift to the input signal.
        drift_rate (float, tuple, list): Rate of the drift applied to the signal.
        func      (str, list) : The type of drift function, either 'linear', 'exp', or a list of these options.
    """
    def __init__(self, p=0.5, drift_rate=0.01, func='linear'):
        self.p = p
        self.drift_rate = drift_rate
        self.func = func

    def _select_value(self, param):
        """Helper function to select a value from param if it's a tuple or list."""
        if isinstance(param, (int, float, str)):
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
        """Apply signal drift to the signal.
        
        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.
        
        Returns:
            Modulated signal with signal drift.
        """
        if np.random.uniform(0, 1) < self.p:
            seed = self.generate_seed()
            np.random.seed(seed)
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]

            # 根据 drift_rate 的类型进行处理
            drift_rate = self._select_value(self.drift_rate)
            func = self._select_value(self.func)

            # Generate a drift signal based on the chosen function
            t = np.linspace(0, 1, sequence_length)
            
            if func == 'linear':
                drift = drift_rate * t
            elif func == 'exp':
                drift = drift_rate * (np.exp(t) - 1)  # Exponential drift
            else:
                raise ValueError("Function type must be 'linear' or 'exp'")
            
            # Apply the drift to all channels
            for i in range(input_dim):
                signal_[:, i] = signal_[:, i] + drift
            
            return signal_
        return signal
