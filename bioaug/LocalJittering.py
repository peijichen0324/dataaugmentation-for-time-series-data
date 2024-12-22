import numpy as np
import hashlib
import time


class LocalJittering(object):
    """Add localized high-frequency jitter to the input time-series data randomly with a given probability.

    Args:
        p            (float)  : Probability of applying jitter to the input signal.
        alpha        (float, tuple, list)  : Scale factor for the high-frequency noise.
        frequency    (int, tuple, list)    : Frequency of the jitter noise.
        duration     (int, tuple, list)    : Duration (in time steps) of each jitter event.
        num_jitters  (int, tuple, list)    : Number of jitter events to add to the signal.
    """
    def __init__(self, p=0.5, alpha=0.5, frequency=50, duration=20, num_jitters=1):
        self.p = p
        self.alpha = alpha
        self.frequency = frequency
        self.duration = duration
        self.num_jitters = num_jitters

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
        """signal: [sequence_length, input_dim]
           signal = signal + α * high_frequency_noise
        """
        if np.random.uniform(0, 1) < self.p:
            seed = self.generate_seed()
            signal_ = np.array(signal).copy()
            np.random.seed(seed)
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]

            # select parameters
            alpha = self._select_value(self.alpha)
            frequency = self._select_value(self.frequency)
            duration = int(self._select_value(self.duration))  # Ensure duration is an integer
            num_jitters = int(self._select_value(self.num_jitters))  # Ensure num_jitters is an integer

            for _ in range(num_jitters):
                # Randomly choose a start point for the jitter
                start_point = np.random.randint(0, sequence_length - duration)
                end_point = start_point + duration
                
                # Generate high-frequency noise for each channel
                for i in range(input_dim):
                    t = np.arange(duration)
                    high_freq_noise = np.sin(2 * np.pi * frequency * t / duration)
                    high_freq_noise += np.random.normal(0, 1, duration)  # Add randomness
                    # Normalize noise
                    high_freq_noise = high_freq_noise / np.max(np.abs(high_freq_noise))
                    # Apply the jitter to the selected segment of the signal
                    signal_[start_point:end_point, i] += alpha * high_freq_noise
            return signal_
        return signal
