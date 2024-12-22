import numpy as np

class CutMix(object):
    """Apply CutMix augmentation to a batch of time-series data.

    Args:
        alpha (float): The parameter for the Beta distribution from which the mixing factor is sampled.
        p     (float): Probability of applying CutMix to the input batch.
        seed  (int)  : A seed value for the random number generator to ensure reproducibility.
    """

    def __init__(self, alpha=1.0, p=0.5, seed=42):
        self.p = p
        self.alpha = alpha
        self.seed = seed

    def __call__(self, batch):
        """Apply CutMix augmentation to the input batch.

        Args:
            batch (tuple): A tuple containing a batch of inputs and labels (inputs, labels).

        Returns:
            tuple: Mixed inputs and mixed labels, or the original inputs and labels if CutMix is not applied.
        """
        x, y = batch
        if np.random.rand() < self.p:
            np.random.seed(self.seed)
            lam = np.random.beta(self.alpha, self.alpha)

            batch_size, seq_len, _ = x.shape
            cut_len = int(seq_len * (1 - lam))

            # Ensure cut_len is not zero to perform a meaningful cut
            if cut_len > 0:
                # Randomly choose the start point for the cut
                cut_start = np.random.randint(0, seq_len - cut_len + 1)

                # Generate a random index for mixing
                indices = np.random.permutation(batch_size)
                x_b = x[indices].copy()
                y_b = y[indices].copy()

                # Debugging: Print out important values
                print(f"cut_len: {cut_len}, cut_start: {cut_start}, lam: {lam}")

                # Replace the region in a copied version of the original sequence with the region from another sequence
                x_mix = x.copy()
                x_mix[:, cut_start:cut_start + cut_len, :] = x_b[:, cut_start:cut_start + cut_len, :]

                # Adjust lambda based on the length of the cut
                lam = 1 - (cut_len / seq_len)
                y_mix = lam * y + (1 - lam) * y_b

                return x_mix, y_mix
        return x, y
