from typing import Protocol, Optional

import numpy as np

class PairedTest(Protocol):

    __name__: str

    def __call__(self, differences: np.ndarray) -> float:
        """
        Calculates and returns p-value for a test.

        differences: np.ndarray
            array of differences of paired observations

        returns: float
            p-value for the given data
        """
        ...

class DiffSampler(Protocol):

    __name__: str

    def __call__(self, n_pairs: int) -> np.ndarray:
        """
        Samples differences under H1.

        n_pairs: int
            number of paired observations to sample

        returns:  np.ndarray of shape (n_pairs,)
            differences sampled from a distribution.
        """
        ...