"""Contains statistics related code."""

import textwrap
from collections.abc import Sequence

import numpy as np
import scipy.stats as st


class Summary:
    """Holds statistical information about supplied dataset."""

    BOOTSTRAP_COUNT = 1000

    def __init__(self, _v: Sequence[float], /, confidence: float = 0.99) -> None:
        """Initialize a Summary objects for the supplied dataset.

        Args:
            _v (Sequence[float]): _description_
            confidence (float, optional): _description_. Defaults to 0.99.

        """
        count = len(_v)
        mean = st.tmean(_v)
        var = st.tvar(_v)
        ci = self._bootstrap_ci(_v, confidence=confidence)
        # abs used because we can't trust that -0.0 won't occur with 1 replicate.
        ci_mean_proportion = (abs(mean - ci[0]) / mean, abs(ci[1] - mean) / mean)
        total = sum(_v)

        self.count: int = count
        self.confidence: float = confidence
        self.mean: float = mean
        self.var: float = var
        self.ci: tuple[float, float] = ci
        self.ci_percent: tuple[float, float] = ci_mean_proportion
        self.total: float = total

    def to_dict(self) -> dict[str, float]:
        """Return dict of low-level stats."""
        return {
            "count": self.count,
            "confidence": self.confidence,
            "mean": self.mean,
            "variance": self.var,
            "ci_lower": self.ci[0],
            "ci_upper": self.ci[1],
            "ci_frac_of_mean_upper": self.ci_percent[0],
            "ci_frac_of_mean_lower": self.ci_percent[1],
            "total": self.total,
        }

    def to_pretty_str(self) -> str:
        """Return interesting statistics in a pretty format."""
        alpha_rep = f"[alpha={1 - self.confidence:.2e}]"
        lower_ci_rep = f"-{self.ci_percent[0]:.2%}"
        upper_ci_rep = f"+{self.ci_percent[1]:.2%}"
        ci_percent_rep = f"({lower_ci_rep}, {upper_ci_rep})"
        return textwrap.dedent(
            f"""
            TOTAL: {self.total:.2f}
            MEAN: {self.mean:.2f}
            CI RATIO {alpha_rep}: {ci_percent_rep}
            """,
        ).strip()

    @staticmethod
    def _lognorm_mean(_v: Sequence[float]) -> float:
        """Compute mean of log-normal fit to supplied data.

        Time durations are all greater than zero, so they come from a skewed
        distribution which cannot ever produce negative values. We can't know
        the true distribution for certain, but log-normal is a good guess.
        """
        s_boot, _, scale_boot = st.lognorm.fit(_v, floc=0)
        return scale_boot * np.exp(0.5 * s_boot**2)

    @staticmethod
    def _bootstrap_ci(_v: Sequence[float], /, confidence: float) -> tuple[float, float]:
        """Use bootstrap approach to compute confidence interval of mean.

        Got some help from Gemini on this one. This function computes the
        confidence interval of the mean using a bootstrap approach.

        A better approach would be using a Bayesian approach. From some light
        researcher, I understand using a Jeffrey prior with a probabilistic
        Bayesian sampling approach would be more accurate, but it's unlikely to
        be worth the effort to understand and apply, with minimal changes in
        values.
        """
        bootstrap_means = []
        n = len(_v)
        rng = np.random.default_rng()
        for _ in range(Summary.BOOTSTRAP_COUNT):
            indices = rng.choice(n, size=n, replace=True)
            sample = [_v[i] for i in indices]
            bootstrap_means.append(Summary._lognorm_mean(sample))

        # Calculate the confidence interval from the bootstrapped means
        alpha = 1 - confidence
        lower_bound = np.percentile(bootstrap_means, (alpha / 2) * 100).item()
        upper_bound = np.percentile(bootstrap_means, (1 - (alpha / 2)) * 100).item()
        return (lower_bound, upper_bound)
