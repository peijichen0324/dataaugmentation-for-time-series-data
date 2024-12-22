import numpy as np
import matplotlib.pyplot as plt


class MixUp(object):
    """Apply MixUp augmentation to the input data.

    Args:
        alpha (float): The parameter for the Beta distribution from which the mixing factor is sampled.
        p     (float): Probability of applying MixUp to the input batch.
        seed  (int)  : A seed value for the random number generator to ensure reproducibility.
    """
    def __init__(self, p=0.5, alpha=0.2, seed=42):
        self.p = p
        self.alpha = alpha
        self.seed = seed

    def __call__(self, batch):
            """Apply MixUp augmentation to the input batch.
            
            Args:
                batch (tuple): A tuple containing a batch of inputs and labels (inputs, labels).
            
            Returns:
                tuple: Mixed inputs and mixed labels, or the original inputs and labels if MixUp is not applied.
            """
            inputs, labels = batch
            if np.random.rand() < self.p:
                np.random.seed(self.seed)
                lam = np.random.beta(self.alpha, self.alpha)
                
                batch_size = inputs.shape[0]
                indices = np.random.permutation(batch_size)
                
                mixed_inputs = lam * inputs + (1 - lam) * inputs[indices, :]
                labels_a, labels_b = labels, labels[indices]
                mixed_labels = lam * labels_a + (1 - lam) * labels_b
                
                return mixed_inputs, mixed_labels
            else:
                return inputs, labels
