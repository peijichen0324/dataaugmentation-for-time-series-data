import numpy as np
import hashlib
import time


class ImpedanceVariation(object):
    """Simulate impedance variation on a physiological signal using trigonometric, exponential, or linear functions.

    Args:
        p        (float)                : Probability of applying impedance variation to the input signal.
        amplitude (float, tuple, list)  : Amplitude of the modulation.
        frequency (float, tuple, list)  : Frequency of the modulation (for trigonometric and exponential functions).
        func     (str, list)            : Type of function to use ('sin', 'exp', 'linear').
        seed     (int)                  : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, amplitude=0.1, frequency=0.01, func='sin'):
        self.p = p
        self.amplitude = amplitude
        self.frequency = frequency
        self.func = func

    def _select_value(self, param):
        """Helper function to select a value from param if it's a tuple or list."""
        if isinstance(param, (int, float)) or isinstance(param, str):
            return param
        elif isinstance(param, tuple) and len(param) == 2:
            return np.random.uniform(param[0], param[1])
        elif isinstance(param, list):
            return np.random.choice(param)
        else:
            raise ValueError("Parameter must be an int, float, str, tuple of length 2, or list.")
        
    @staticmethod
    def generate_seed():
        current_time = str(time.time() * 1000)
        hash_object = hashlib.md5(current_time.encode())
        hash_integer = int(hash_object.hexdigest(), 16)
        random_integer = hash_integer % 100
        return random_integer

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]
           sin: signal = signal + α * high_frequency_noise
           exp: 
           linear:
        """
        if np.random.uniform(0, 1) < self.p:
            seed = self.generate_seed()

            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]

            # Select parameters
            amplitude = self._select_value(self.amplitude)
            frequency = self._select_value(self.frequency)
            func = self._select_value(self.func)

            # Generate a modulation signal based on the chosen function
            t = np.linspace(0, 1, sequence_length)
            
            if func == 'sin':
                modulation = 1 + amplitude * np.sin(2 * np.pi * frequency * t)
            elif func == 'exp':
                modulation = 1 + amplitude * np.exp(-t * frequency)
            elif func == 'linear':
                modulation = 1 + amplitude * t
            else:
                raise ValueError("Function type must be 'sin', 'cos', 'exp', or 'linear'")
            
            # Apply the modulation to all channels
            for i in range(input_dim):
                signal_[:, i] = signal_[:, i] * modulation
            
            return signal_
        return signal
