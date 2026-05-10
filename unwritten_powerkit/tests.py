"""
Объекты тестов, которые мы хотим проанализировать. Все тесты должны следовать протоколу interfaces.PairedTest 
"""


from typing import Optional
import numpy as np
from scipy import stats 
from interfaces import PairedTest



def paired_ttest(diffs: np.ndarray) -> float:
    """Two-sided paired t-test."""
    # TODO: implement using scipy.stats 
    res = stats.ttest_1samp(diffs, popmean=0)
    return res.pvalue


def wilcoxon_signed_rank(diffs: np.ndarray) -> float:
    """Two-sided Wilcoxon signed-rank test."""
    # TODO: implement using scipy.stats 
    res = stats.wilcoxon(diffs)
    return res.pvalue

class PermutationTest:
    """
    Paired permutation test.

    Under H0 the sign of each difference is equally likely to be + or -.
    The test statistic is the mean of the absolute differences.
    """

    __name__ = 'Paired permutation test'

    def __init__(self, rng: Optional[np.random.Generator] = None, n_permutations: int = 10000):
        self.rng = rng if rng is not None else np.random.default_rng()
        self.n_permutations = n_permutations

    def __call__(self, differences: np.ndarray) -> float:
        
        # TODO: implement yourself 
        t_obs = np.abs(np.mean(differences)) # Тут в задании написана беллиберда, тк при такой постановке статистики у нас ничего не будет меняться при пермутации
        # поэтому я предположил что там должен был быть другой порядок: не среднее абсолютных, а абсолютное среднего

        signs = self.rng.choice(
            [-1, 1], 
            size=(self.n_permutations, len(differences))
        )

        permuted_means = (signs * differences).mean(axis=1)
        t_perm = np.abs(permuted_means)

        p_value = (np.sum(t_perm >= t_obs) + 1) / (self.n_permutations + 1) # на всяий случай от нулей в числитееле или знаменателе
        
        return p_value
