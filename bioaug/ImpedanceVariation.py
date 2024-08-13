import numpy as np
import matplotlib.pyplot as plt

class ImpedanceVariation(object):
    """Simulate impedance variation on a physiological signal using trigonometric, exponential, or linear functions.

    Args:
        p        (float) : Probability of applying impedance variation to the input signal.
        amplitude (float) : Amplitude of the modulation.
        frequency (float) : Frequency of the modulation (for trigonometric and exponential functions).
        func     (str)   : Type of function to use ('sin', 'cos', 'exp', 'linear').
        seed     (int)   : A seed value for the random number generator.
    """
    def __init__(self, p=0.5, amplitude=0.1, frequency=0.01, func='sin', seed=42):
        self.p = p
        self.amplitude = amplitude
        self.frequency = frequency
        self.func = func
        self.seed = seed

    def __call__(self, signal):
        """signal: [sequence_length, input_dim]
           sin: signal = signal + α * high_frequency_noise
           exp: 
           linear:
        """
        if np.random.uniform(0, 1) < self.p:
            np.random.seed(self.seed)
            signal_ = np.array(signal).copy()
            sequence_length, input_dim = signal_.shape[0], signal_.shape[1]
            
            # Generate a modulation signal based on the chosen function
            t = np.linspace(0, 1, sequence_length)
            
            if self.func == 'sin':
                modulation = 1 + self.amplitude * np.sin(2 * np.pi * self.frequency * t)
            elif self.func == 'exp':
                modulation = 1 + self.amplitude * np.exp(-t * self.frequency)
            elif self.func == 'linear':
                modulation = 1 + self.amplitude * t
            else:
                raise ValueError("Function type must be 'sin', 'cos', 'exp', or 'linear'")
            
            # Apply the modulation to all channels
            for i in range(input_dim):
                signal_[:, i] = signal_[:, i] * modulation
            
            return signal_
        return signal

# Example usage
if __name__ == '__main__':
    data = np.random.normal(loc=1, scale=1, size=(500, 6))
    gn = ImpedanceVariation(p=1.0, amplitude=1, frequency=0.1, func='sin')
    aug_data = gn(data)

    raw_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = raw_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), data[:, plt_index-1])

    aug_fig = plt.figure(figsize=(5, 5))
    for plt_index in range(1, 7):
        ax = aug_fig.add_subplot(3, 2, plt_index)
        ax.plot(list(range(500)), aug_data[:, plt_index-1], color='r')
    plt.show()
