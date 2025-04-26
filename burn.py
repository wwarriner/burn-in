"""Contains code to run replicates of matmul on all CPUs and GPUs.

documentation
"""

from __future__ import annotations

import abc
import logging
import multiprocessing as mp
import os
import textwrap
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from multiprocessing.pool import Pool as mpPool  # type hint
from types import MappingProxyType

import numpy as np
import scipy.stats as st
import torch as t
import torch.multiprocessing as tmp
from torch.multiprocessing.pool import Pool as tmpPool  # type hint

import conf

LOG_LEVEL = logging.INFO
logging.basicConfig(
    filename="output.log",
    filemode="a",
    encoding="utf-8",
    level=LOG_LEVEL,
    format="%(levelname)s:%(asctime)s:%(funcName)s: %(message)s",
)
LOG = logging.getLogger("burn")

type NullFunction = Callable[[], None]
type Pool = tmpPool | mpPool


class Summary:
    """Holds statistical information about supplied dataset."""

    BOOTSTRAP_COUNT = 1000

    def __init__(self, _v: Sequence[float], /, confidence: float = 0.99) -> None:
        """Initialize a Summary objects for the supplied dataset.

        Args:
            _v (Sequence[float]): _description_
            confidence (float, optional): _description_. Defaults to 0.99.

        """
        mean = st.tmean(_v)
        var = st.tvar(_v)
        ci = self._bootstrap_ci(_v, confidence=confidence)
        total = sum(_v)

        self.confidence: float = confidence
        self.mean: float = mean
        self.var: float = var
        self.ci: tuple[float, float] = ci
        self.ci_percent: tuple[float, float] = ci_mean_proportion
        self.total: float = total

    def to_pretty_str(self) -> str:
        """Return interesting statistics in a pretty format."""
        alpha_rep = f"[alpha={1 - self.confidence:.2e}]"
        # abs used because we can't trust that -0.0 won't occur with 1 replicate.
        lower_ci_rep = f"-{abs(self.ci_percent[0]):.2%}"  # abs to avoid "--0.00"
        upper_ci_rep = f"+{abs(self.ci_percent[1]):.2%}"  # abs to avoid "+-0.00"
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


class Burn(abc.ABC):
    """Abstract Burn class.

    Has concrete implementation of burn method. Subclasses must supply
    implementations of other methods.
    """

    @property
    @abc.abstractmethod
    def _device_type_singular(self) -> str: ...

    @property
    @abc.abstractmethod
    def _device_type_plural(self) -> str: ...

    def burn(self, device_config: dict) -> list[Summary]:
        """Run matmul on all instances of the associated device type."""
        with self._make_pool_context() as pool:
            results = pool.starmap(
                self._run_on_device,
                [[device_config]] * self._get_device_count(),
            )

        LOG.info(
            "%s results across %d %s",
            self._device_type_singular,
            self._get_device_count(),
            self._device_type_plural,
        )
        LOG.info(
            "matrix_size: %d, replicates: %d",
            device_config["matrix_size"],
            device_config["replicates"],
        )
        for result in results:
            LOG.info("%s", result.to_pretty_str())

        return results

    def _run_on_device(
        self,
        device_config: dict,
    ) -> Summary:
        matrix_size = device_config["matrix_size"]
        replicate_count = device_config["replicates"]

        device = self._get_device()
        matmul_fn = _to_null_function(
            t.matmul,
            (_square_randn(matrix_size, device), _square_randn(matrix_size, device)),
        )
        _warmup(matmul_fn)
        times = [self._timeit(matmul_fn) for _ in range(replicate_count)]
        return Summary(times)

    @abc.abstractmethod
    def _get_device_count(self) -> int: ...

    @abc.abstractmethod
    def _get_device(self) -> int | str: ...

    @abc.abstractmethod
    def _make_pool_context(self) -> Pool: ...

    @abc.abstractmethod
    def _timeit(self, _fn: NullFunction) -> float: ...


class CpuBurn(Burn):
    """CPU implementation of Burn class."""

    @property
    def _device_type_singular(self) -> str:
        return "cpu"

    @property
    def _device_type_plural(self) -> str:
        return "cpus"

    def _get_device(self) -> str:
        return "cpu"

    def _get_device_count(self) -> int:
        if "SLURM_CPUS_ON_NODE" in os.environ:
            out = int(os.environ["SLURM_CPUS_ON_NODE"])
        else:
            try:
                out = mp.cpu_count()
            except NotImplementedError:
                out = 0

        return out

    def _make_pool_context(self) -> Pool:
        return mp.Pool(self._get_device_count())

    def _timeit(self, _fn: NullFunction) -> float:
        """Run the supplied function once and return time in seconds."""
        start_ns = time.perf_counter_ns()
        _fn()
        stop_ns = time.perf_counter_ns()

        return (stop_ns - start_ns) / 1_000_000_000  # ns to s


class GpuBurn(Burn):
    """GPU implementation of Burn class."""

    @property
    def _device_type_singular(self) -> str:
        return "gpu"

    @property
    def _device_type_plural(self) -> str:
        return "gpus"

    def _get_device(self) -> int | str:
        return t.cuda.current_device()

    def _get_device_count(self) -> int:
        return t.cuda.device_count() if t.cuda.is_available() else 0

    def _make_pool_context(self) -> Pool:
        return tmp.Pool(self._get_device_count())

    def _timeit(self, _fn: NullFunction) -> float:
        """Run the supplied function once and return time in seconds."""
        start_ms = t.cuda.Event(enable_timing=True)
        stop_ms = t.cuda.Event(enable_timing=True)
        t.cuda.synchronize()
        start_ms.record()  # type: ignore reportCallIssue
        _fn()
        stop_ms.record()  # type: ignore reportCallIssue
        t.cuda.synchronize()

        return (start_ms.elapsed_time(stop_ms)) / 1_000  # ms to s


def _square_randn(size: int, /, device: int | str) -> t.Tensor:
    if size <= 0:
        size = 1

    shape = [size, size]

    return t.randn(shape, device=device)


def _to_null_function(
    _fn: Callable,
    args: Iterable = (),
    kwargs: Mapping = MappingProxyType({}),
) -> NullFunction:
    """Return a function with the arguments applied and absorbing the return value.

    Returned function is null-adic and null-ary.
    """

    def _null_fn() -> None:
        _fn(*args, **kwargs)

    return _null_fn


def _warmup(_fn: NullFunction) -> None:
    for _ in range(10):
        _fn()


def burn(config: dict) -> None:
    """Perform execution with supplied config."""
    LOG.info("execution started")
    CpuBurn().burn(config["computation"]["cpu"])
    GpuBurn().burn(config["computation"]["gpu"])
    LOG.info("execution stopped")


if __name__ == "__main__":
    LOG.info("program started")
    args = conf.get_args()
    config = conf.load_or_build_config(args.config_file)
    LOG.info("program stopped")
