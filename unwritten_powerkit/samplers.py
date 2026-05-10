"""
Генераторы, которые мы хотим проанализировать. Все генераторы должны следовать протоколу interfaces.DiffSampler 
"""

from typing import Optional
import numpy as np

from interfaces import DiffSampler 

class NormalShiftSampler:
    """
    Diffs ~ Normal(effect_size, sigma^2).

    effect_size: mean of the paired difference (Cohen's d = effect_size / sigma).
    """

    def __init__(self, effect_size: float, sigma: float,
                 rng: Optional[np.random.Generator] = None):
        self.effect_size = effect_size
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.default_rng()
        self.__name__ = f"Normal(δ={effect_size}, σ={sigma})"

    def __call__(self, n_pairs: int) -> np.ndarray:
        return self.rng.normal(loc=self.effect_size, scale=self.sigma,
                               size=n_pairs)


class LognormalShiftSampler:
    """
    Diffs ~ LogNormal(mu, sigma) - median(LogNormal).
    """
    def __init__(self, mu: float, sigma: float,
                 rng: Optional[np.random.Generator] = None):
        self.mu = mu
        self.sigma = sigma
        self.rng = rng if rng is not None else np.random.default_rng()
        self.__name__ = f"LogNormal(μ={mu}, σ={sigma})"

    def __call__(self, n_pairs: int) -> np.ndarray:
        median = np.exp(self.mu)
        return self.rng.lognormal(mean=self.mu, sigma=self.sigma,
                                  size=n_pairs) - median


class MixtureSampler:
    """
    Contaminated normal:
        (1 - contamination) * N(effect_size, sigma^2)
        + contamination * N(0, (outlier_scale * sigma)^2)

    Models a realistic distribution with occasional large outliers.
    The bulk of observations show a true shift; contaminating observations
    are zero-centered noise with inflated variance.
    """
    def __init__(
        self,
        effect_size: float,
        sigma: float = 1.0,
        contamination: float = 0.1,
        outlier_scale: float = 5.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.effect_size = effect_size
        self.sigma = sigma
        self.contamination = contamination
        self.outlier_scale = outlier_scale
        self.rng = rng if rng is not None else np.random.default_rng()
        self.__name__ = (
            f"Mixture(δ={effect_size}, σ={sigma}, contam={contamination})"
        )

    def __call__(self, n_pairs: int) -> np.ndarray:
        base = self.rng.normal(loc=self.effect_size, scale=self.sigma, size=n_pairs)
        outliers = self.rng.normal(loc=0, scale=self.outlier_scale * self.sigma, size=n_pairs)
        mask = self.rng.random(size=n_pairs) < self.contamination
        return np.where(mask, outliers, base)

class NegativeOutlierMixtureSampler:
    """
    Mixture of two normal distributions far from the true mean difference.

    The second component mean is chosen so that E[D] = true_delta.
    """

    def __init__(
        self,
        true_delta: float = 0.5,
        p_left: float = 0.5,
        distance: float = 4.0,
        left_sigma: float = 0.5,
        right_sigma: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ):
        if not 0 < p_left < 1:
            raise ValueError("p_left must be between 0 and 1")

        self.true_delta = true_delta
        self.p_left = p_left
        self.distance = distance
        self.left_sigma = left_sigma
        self.right_sigma = right_sigma
        self.rng = rng if rng is not None else np.random.default_rng()

        self.left_mean = self.true_delta - self.distance
        self.right_mean = (
            self.true_delta - self.p_left * self.left_mean
        ) / (1 - self.p_left)

        self.__name__ = (
            f"TwoNormalMixture("
            f"δ={true_delta}, left={self.left_mean:.2f}, "
            f"right={self.right_mean:.2f}, p_left={p_left})"
        )

    def __call__(self, n_pairs: int) -> np.ndarray:
        is_left = self.rng.random(n_pairs) < self.p_left
        differences = np.empty(n_pairs)

        differences[is_left] = self.rng.normal(
            loc=self.left_mean,
            scale=self.left_sigma,
            size=is_left.sum(),
        )

        differences[~is_left] = self.rng.normal(
            loc=self.right_mean,
            scale=self.right_sigma,
            size=(~is_left).sum(),
        )

        return differences
