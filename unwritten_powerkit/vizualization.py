import matplotlib
import matplotlib.pyplot as plt
import matplotlib.axes
from matplotlib.lines import Line2D
import numpy as np

from core import PowerResult, clopper_pearson_ci, ComparisonSpec


# Visual style constants
_COLORS  = ["#2196F3", "#E91E63", "#4CAF50"]
_MARKERS = ["o", "s", "^"]
_CI_ALPHA = 0.15   # transparency of the confidence band fill


def plot_power_curve_on_ax(
    ax: matplotlib.axes.Axes,
    results_for_sampler: list[list[PowerResult]],   # [test_idx][n_idx]
    sample_sizes: list[int],
    test_names: list[str],
    alpha: float = 0.05,
    ci_level: float = 0.95,
    show_legend: bool = True,
) -> None:
    """
    Draw power curves with Clopper-Pearson CI bands onto a single Axes.

    Parameters
    ----------
    ax                    : the Axes to draw on
    results_for_sampler   : results[test_idx][n_idx]
    sample_sizes          : x-axis values
    test_names            : display names, one per test
    alpha                 : significance level (drawn as a reference line)
    ci_level              : two-sided level for the CI bands (default 95%)
    show_legend           : whether to draw a legend on this Axes
    """
    for ti, (curve, tname) in enumerate(zip(results_for_sampler, test_names)):
        color  = _COLORS[ti % len(_COLORS)]
        marker = _MARKERS[ti % len(_MARKERS)]

        powers = np.array([r.power for r in curve])
        lo     = np.array([clopper_pearson_ci(r.n_rejections, r.n_simulations, ci_level)[0] for r in curve])
        hi     = np.array([clopper_pearson_ci(r.n_rejections, r.n_simulations, ci_level)[1] for r in curve])

        ax.plot(sample_sizes, powers, color=color, marker=marker,
                linewidth=2, markersize=6, label=tname)
        ax.fill_between(sample_sizes, lo, hi, color=color, alpha=_CI_ALPHA)

    ax.axhline(0.8,  color="gray",  linestyle="--", linewidth=1, label="80% power")
    ax.axhline(alpha, color="black", linestyle=":",  linewidth=1, label=f"α={alpha}")
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("n (pairs)")

    if show_legend:
        ax.legend(fontsize=8)


def plot_comparison(
    spec: ComparisonSpec,
    results: list[list[list[PowerResult]]],
    ci_level: float = 0.95,
    figsize_per_panel: tuple[float, float] = (5.0, 4.0),
    suptitle: str = "Power Analysis: Paired Tests",
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Plot one panel per sampler, each showing power curves for all tests.

    Parameters
    ----------
    spec              : ComparisonSpec used to produce the results
    results           : results[test_idx][sampler_idx][n_idx]
    ci_level          : two-sided CI level for the Clopper-Pearson bands
    figsize_per_panel : (width, height) of each subplot panel
    suptitle          : overall figure title
    save_path         : if given, save the figure to this path
    """

    # TODO: отрисовать графики кривых мощностей тестов в разных условиях, опираясь на данные в spec и results.
    # Как именно визуализировать -- решайте сами, но помните, что ваши графики должны быть оформлены аккуратно, и хорошо читаться
    # В этом случае разумно сравнивать кривые в одних условиях.

    # Жесткие требования: отразить кривые и их доверительные интервалы всех тестов для всех сэмплеров во всех точках 

    # Не забывайте переиспользовать код в этом файле и других файлах проекта.  
    n_tests = len(spec.tests)
    n_samplers = len(spec.samplers)
    
    n_cols = min(3, n_samplers)
    n_rows = int(np.ceil(n_samplers / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(figsize_per_panel[0] * n_cols, figsize_per_panel[1] * n_rows),
        sharex=True,
        sharey=True,
    )
    axes_flat = np.atleast_1d(axes).ravel()

    test_names = [getattr(test, "__name__", repr(test)) for test in spec.tests]

    for si, sampler in enumerate(spec.samplers):
        ax = axes_flat[si]
        results_for_sampler = [results[ti][si] for ti in range(n_tests)]

        plot_power_curve_on_ax(
            ax=ax,
            results_for_sampler=results_for_sampler,
            sample_sizes=spec.sample_sizes,
            test_names=test_names,
            alpha=spec.alpha,
            ci_level=ci_level,
            show_legend=False,
        )

        ax.set_title(getattr(sampler, "__name__", repr(sampler)))
        if si % n_cols == 0:
            ax.set_ylabel("Power")


    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(len(labels), 5),
        frameon=False,
    )

    fig.suptitle(suptitle)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig



def print_comparison(
    spec: ComparisonSpec,
    results: list[list[list[PowerResult]]],
) -> None:
    """Pretty-print power curves with Clopper-Pearson CIs."""
    col_w = 22
    header = f"{'n':>6}" + "".join(
        f"  {getattr(t, '__name__', '?'):>{col_w}}" for t in spec.tests
    )

    for si, sampler in enumerate(spec.samplers):
        sname = getattr(sampler, "__name__", repr(sampler))
        print(f"\n{'='*80}")
        print(f"H1 distribution : {sname}")
        print(f"alpha={spec.alpha}, simulations={spec.n_simulations}")
        print("=" * 80)
        print(header)
        print("-" * len(header))
        for ni, n in enumerate(spec.sample_sizes):
            row = f"{n:>6}"
            for ti in range(len(spec.tests)):
                r = results[ti][si][ni]
                lo, hi = clopper_pearson_ci(r.n_rejections, r.n_simulations)
                cell = f"{r.power:.3f} [{lo:.3f},{hi:.3f}]"
                row += f"  {cell:>{col_w}}"
            print(row)