import numpy as np
import hashlib
import time


class RandomCutout(object):
    """Random cutout selected area of the input time-series data randomly with a given probability.
    
    Args:
        p        (float) : probability of the sensor data to be performed the random cutout. Default value is 0.5.
        area     (float, tuple, list) : Size of the cutout area.
        num      (int, tuple, list)   : Number of cutout areas.
        default  (float, tuple, list) : Replace value of the cutout area.
    """

    def __init__(self, p=0.5, area=25, num=4, default=0.0):
        self.p = p
        self.area = area
        self.num = num
        self.default = default

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
            sequence_length = signal.shape[0]
            mask = np.ones(sequence_length, np.float32)
            np.random.seed(seed)

            # Select parameters
            area = int(self._select_value(self.area))  # Ensure area is an integer
            num = int(self._select_value(self.num))  # Ensure num is an integer
            default = int(self._select_value(self.default))

            # Apply the cutout
            for _ in range(num):
                x = np.random.randint(sequence_length)
                x1 = np.clip(x - area // 2, 0, sequence_length)
                x2 = np.clip(x + area // 2, 0, sequence_length)
                mask[x1:x2] = 0
            
            # Apply the mask to all channels
            mask = np.expand_dims(mask, axis=1)  # Ensure mask has the right shape [sequence_length, 1]
            mask_b = 1 - mask  # Inverse mask for the cutout
            
            # Apply the mask to the signal
            signal_ = signal_ * mask + mask_b * default
            return signal_
        return signal
