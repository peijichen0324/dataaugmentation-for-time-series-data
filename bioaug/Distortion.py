import numpy as np
import hashlib
import time


class Distortion(object):
    """Simulate nonlinear distortion by applying a nonlinear transformation to the input time-series data.

    Args:
        p                (float) : Probability of applying nonlinear distortion to the input signal.
        distortion_degree (float, tuple, list): Degree of nonlinearity applied to the signal (n > 1).
        harmonic_degree   (float, tuple, list): Degree of harmonic distortion added to the signal.
        phase_shift       (float, tuple, list): Degree of phase distortion (in radians).
        distortion_type   (str, list)         : Type of distortion to apply ('amplitude', 'harmonic', 'phase').
    """

    def __init__(self, p=0.5, distortion_degree=2.0, harmonic_degree=0.1, phase_shift=0.1,
                 distortion_type='amplitude'):
        self.p = p
        self.distortion_degree = distortion_degree
        self.harmonic_degree = harmonic_degree
        self.phase_shift = phase_shift
        self.distortion_type = distortion_type

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

    def apply_amplitude_distortion(self, signal):
        """Apply amplitude distortion to the signal."""
        distortion_degree = self._select_value(self.distortion_degree)
        return np.sign(signal) * np.abs(signal) ** distortion_degree

    def apply_harmonic_distortion(self, signal):
        """Apply harmonic distortion to the signal."""
        harmonic_degree = self._select_value(self.harmonic_degree)
        harmonic_signal = harmonic_degree * np.sin(2 * np.pi * np.linspace(0, 1, signal.shape[0]) * 5)
        return signal + harmonic_signal[:, np.newaxis]

    def apply_phase_distortion(self, signal):
        """Apply phase distortion by altering the phase of the signal's frequency components."""
        # Perform Fourier transform to move signal to the frequency domain
        phase_shift = self._select_value(self.phase_shift)
        signal_fft = np.fft.fft(signal, axis=0)
        # Create a phase shift array that applies the phase shift to each frequency component
        phase_shifts = np.exp(1j * phase_shift * np.linspace(0, signal.shape[0] - 1, signal.shape[0]))
        # Apply phase shift to each frequency component
        signal_fft = signal_fft * phase_shifts[:, np.newaxis]
        # Perform inverse Fourier transform to move back to time domain
        signal_phase_distorted = np.fft.ifft(signal_fft, axis=0).real
        return signal_phase_distorted
    
    @staticmethod
    def generate_seed():
        current_time = str(time.time() * 1000)
        hash_object = hashlib.md5(current_time.encode())
        hash_integer = int(hash_object.hexdigest(), 16)
        random_integer = hash_integer % 100
        return random_integer

    def __call__(self, signal):
        """Apply nonlinear distortion to the signal.

        Args:
            signal: [sequence_length, input_dim] - The original physiological signal.

        Returns:
            Distorted signal with nonlinear transformations.
        """
        if np.random.uniform(0, 1) < self.p:
            seed = self.generate_seed()
            np.random.seed(seed)
            signal_ = np.array(signal).copy()
            distortion_type = self._select_value(self.distortion_type)
            # Apply selected type of distortion
            if distortion_type == 'amplitude':
                signal_ = self.apply_amplitude_distortion(signal_)
            elif distortion_type == 'harmonic':
                signal_ = self.apply_harmonic_distortion(signal_)
            elif distortion_type == 'phase':
                signal_ = self.apply_phase_distortion(signal_)
            else:
                raise ValueError("Invalid distortion type. Choose from 'amplitude', 'harmonic', 'phase', or 'all'.")
            return signal_
        return signal
