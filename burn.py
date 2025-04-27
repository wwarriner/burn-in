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
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
import scipy.stats as st
import torch as t
import torch.multiprocessing as tmp

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as mpPool
    from pathlib import PurePath

    from torch.multiprocessing.pool import Pool as tmpPool


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

    def __init__(self) -> None:
        """Initialize a Burn object."""
        self._results: dict[str, Summary] | None = None

    def burn_async(self, device_config: dict, pool: Pool) -> None:
        """Run matmul on all instances of the associated device type.

        Returns asynchronous results.
        """
        self._results = {}
        for index in range(self._get_device_count()):
            args = (
                self._get_device_name(index),
                self._get_device(index),
                device_config,
                self._timeit,
            )
            pool.apply_async(
                self._run_on_device,
                args,
                callback=self._collect_result,
                error_callback=self._log_error,
            )
        LOG.info(
            "%s: running across %d %s",
            self._device_type_singular,
            self._get_device_count(),
            self._device_type_plural,
        )
        LOG.info(
            "matrix_size: %d, replicates: %d",
            device_config["matrix_size"],
            device_config["replicates"],
        )

    def get_results(self) -> dict[str, Summary]:
        """Synchronize asynchronous operations."""
        if self._results is None:
            raise RuntimeError

        results = self._results
        self._results = None
        return results

    def _collect_result(self, result: tuple[str, Summary]) -> None:
        if self._results is None:
            LOG.warning("result was None")
        else:
            device = result[0]
            summary = result[1]
            self._results[device] = summary

    def _log_error(self, exception: BaseException) -> None:
        LOG.exception(exception)

    @abc.abstractmethod
    def create_pool(self) -> Pool:
        """Create a Pool."""
        ...

    @staticmethod
    def _run_on_device(
        device_name: str,
        device: int | str,
        device_config: dict,
        timeit_fn: Callable[[NullFunction], float],
    ) -> tuple[str, Summary]:
        """Run matmul on supplied device.

        Returns device and Summary in tuple.
        """
        matrix_size = device_config["matrix_size"]
        replicate_count = device_config["replicates"]

        matmul_fn = _to_null_function(
            t.matmul,
            (_square_randn(matrix_size, device), _square_randn(matrix_size, device)),
        )
        _warmup(matmul_fn)
        times = [timeit_fn(matmul_fn) for _ in range(replicate_count)]
        return device_name, Summary(times)

    @abc.abstractmethod
    def _get_device_count(self) -> int: ...

    @abc.abstractmethod
    def _get_device(self, index: int) -> int | str: ...

    @abc.abstractmethod
    def _get_device_name(self, index: int) -> str: ...

    @staticmethod
    @abc.abstractmethod
    def _timeit(_fn: NullFunction) -> float: ...


class CpuBurn(Burn):
    """CPU implementation of Burn class."""

    @property
    def _device_type_singular(self) -> str:
        return "cpu"

    @property
    def _device_type_plural(self) -> str:
        return "cpus"

    def _get_device(self, _: int) -> str:
        return "cpu"

    def _get_device_name(self, index: int) -> str:
        return f"cpu:{index}"

    def _get_device_count(self) -> int:
        if "SLURM_CPUS_ON_NODE" in os.environ:
            out = int(os.environ["SLURM_CPUS_ON_NODE"])
        else:
            try:
                out = mp.cpu_count()
            except NotImplementedError:
                out = 0

        return out

    def create_pool(self) -> Pool:
        """Create a CPU pool."""
        return mp.Pool(self._get_device_count())

    @staticmethod
    def _timeit(_fn: NullFunction) -> float:
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

    def _get_device(self, index: int) -> int | str:
        return index

    def _get_device_name(self, index: int) -> str:
        return f"cuda:{index}"

    def _get_device_count(self) -> int:
        return t.cuda.device_count() if t.cuda.is_available() else 0

    def create_pool(self) -> Pool:
        """Create a GPU pool."""
        return tmp.Pool(self._get_device_count())

    @staticmethod
    def _timeit(_fn: NullFunction) -> float:
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


def burn(config: dict) -> dict[str, Summary]:
    """Perform execution with supplied config."""
    cpu = CpuBurn()
    gpu = GpuBurn()
    devices = (cpu, gpu)
    device_configs = (config["computation"]["cpu"], config["computation"]["gpu"])

    LOG.info("execution started")
    with cpu.create_pool() as cpu_pool, gpu.create_pool() as gpu_pool:
        pools = (cpu_pool, gpu_pool)

        for device, pool, device_config in zip(devices, pools, device_configs):
            device.burn_async(device_config, pool)

        for pool in pools:
            pool.close()
            pool.join()

    LOG.info("execution stopped")
    LOG.info("displaying results")
    results = {**(cpu.get_results()), **(gpu.get_results())}
    for device_name, summary in results.items():
        LOG.info("device %s", device_name)
        LOG.info("%s", summary.to_pretty_str())
    LOG.info("results concluded")

    return results


def to_csv(filepath: PurePath, results: dict[str, Summary]) -> None:
    """Write results to CSV file at filepath."""
    if not results:
        return

    devices_sorted = sorted(results.keys())
    stats = {device: results[device].to_dict() for device in devices_sorted}

    stat_names: set[str] = set()
    for stat in stats.values():
        stat_names |= stat.keys()
    stat_names_sorted = sorted(stat_names)

    header_line = ",".join(["device", *stat_names_sorted])
    lines = [header_line]
    for device, stat in stats.items():
        values_sorted = [stat[name] for name in stat_names_sorted]
        values_to_write = [f"{value:.17f}" for value in values_sorted]
        stat_line = ",".join([device, *values_to_write])
        lines.append(stat_line)

    Path(filepath.parent).mkdir(parents=True, exist_ok=True)
    with Path(filepath).open("w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    LOG.info("program started")
    args = conf.get_args()
    config = conf.load_or_build_config(args.config_file)
    results = burn(config)
    to_csv(args.output_file, results)
    LOG.info("program stopped")
