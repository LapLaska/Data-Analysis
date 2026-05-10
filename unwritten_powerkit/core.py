from interfaces import PairedTest, DiffSampler
from dataclasses import dataclass
from scipy import stats 

@dataclass
class PowerResult:
    """
    Class for storing the results of a single run of power simulations
    """
    test_name: str
    sampler_name: str
    n_pairs: int
    alpha: float
    n_simulations: int
    n_rejections: int
    power: float

    def __repr__(self) -> str:
        return (
            f"PowerResult({self.test_name} | {self.sampler_name} | "
            f"n={self.n_pairs} | power={self.power:.3f} "
        )

    def __str__(self) -> str:
        return (
            f"{self.test_name:<35} | {self.sampler_name:<40} | "
            f"n={self.n_pairs:>4} | power={self.power:.3f}"
        )


def clopper_pearson_ci(
    n_rejections: int,
    n_simulations: int,
    ci_level: float = 0.95,
) -> tuple[float, float]:
    """
    Exact Clopper-Pearson confidence interval for a binomial proportion.

    Uses the relationship between the binomial CDF and the beta distribution:
      lower = Beta(ci_level/2;  k,   n-k+1)
      upper = Beta(1-ci_level/2; k+1, n-k)

    Parameters
    ----------
    n_rejections  : number of successes (rejections of H0)
    n_simulations : total number of trials
    ci_level      : two-sided significance level for the CI (default 95% CI)
    """
    alpha = 1 - ci_level

    k, n = n_rejections, n_simulations
    lo = stats.beta.ppf(alpha / 2,       k,     n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2,   k + 1, n - k)     if k < n else 1.0
    return float(lo), float(hi)


def estimate_power(
    test: PairedTest,
    sampler: DiffSampler,
    n_pairs: int,
    alpha: float = 0.05,
    n_simulations: int = 2000,
    seed: int | None = None,
) -> PowerResult:
    """
    Estimate power via Monte Carlo simulation.

    Parameters
    ----------
    test          : PairedTest  — callable (diffs) -> p_value
    sampler       : DiffSampler — callable (n_pairs) -> diffs
    n_pairs       : number of paired observations per simulated dataset
    alpha         : significance level
    n_simulations : number of Monte Carlo replicates
    seed          : random seed (for samplers/tests that don't own their rng)
    """
    
    # TODO: calculate the power estimate for the given test and sampler; use parallelization and efficient numpy operations
    # hint: you can use tqdm library to visualize the progress (and notice if something is stuck rather then running)

    n_rejections = 0

    for i in range(n_simulations):
        data = sampler(n_pairs)
        n_rejections += int(test(data) <= alpha)
   
    power = n_rejections / n_simulations

    return PowerResult(
        test_name=getattr(test, "__name__", repr(test)),
        sampler_name=getattr(sampler, "__name__", repr(sampler)),
        # TODO: finish initialization
        n_pairs=n_pairs,
        alpha=alpha,
        n_simulations=n_simulations,
        n_rejections=n_rejections,
        power=power
        )


def power_curve(
    test: PairedTest,
    sampler: DiffSampler,
    sample_sizes: list[int],
    alpha: float = 0.05,
    n_simulations: int = 2000,
    seed: int | None = None,
) -> list[PowerResult]:
    """
    Estimate power for a range of sample sizes.
    returns: list, where each item is a power estimate for an element in sample_sizes (in the same order)
    """

    # TODO: run an estimation for given points (sample_sizes) of the power curve reusing the code above
    
    powers = [estimate_power(
        test,
        sampler,
        sample_sizes[i],
        alpha,
        n_simulations,
        seed
    ) for i in range(len(sample_sizes))]
    
    
    return powers



@dataclass
class ComparisonSpec:
    """
    Class for storing the configuration of parallel power analysis simulation for several compaired tests 
    """
    tests: list[PairedTest]
    samplers: list[DiffSampler]
    sample_sizes: list[int]
    alpha: float = 0.05
    n_simulations: int = 2000
    seed: int | None = 42


def run_comparison(spec: ComparisonSpec) -> list[list[list[PowerResult]]]:
    """
    Run power curves for every (test, sampler) pair.

    Return format: results[test_index][sampler_index] -> list[PowerResult], a power curve for given test and sampler
    """

    # TODO: run comparison reusing the code above

    results = [[0] * len(spec.samplers) for _ in range(len(spec.tests))]

    for test_ind in range(len(spec.tests)):
        for sampler_ind in range(len(spec.samplers)):
            results[test_ind][sampler_ind] = power_curve(
                spec.tests[test_ind],
                spec.samplers[sampler_ind],
                spec.sample_sizes,
                spec.alpha,
                spec.n_simulations,
                spec.seed
            )

    return results