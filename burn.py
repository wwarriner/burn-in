"""Contains code to run replicates of matmul on all CPUs and GPUs.

documentation
"""

from __future__ import annotations

import abc
import contextlib
import logging
import multiprocessing as mp
import os
import textwrap
import time

import numpy as np
import scipy.stats as st
import torch as t
import torch.multiprocessing as tmp

import conf
from fileops import to_csv
from stats import Summary

if TYPE_CHECKING:
    from multiprocessing.pool import Pool as mpPool

    from torch.multiprocessing.pool import Pool as tmpPool

LOG = logging.getLogger("burn")

type NullFunction = Callable[[], None]
type Pool = tmpPool | mpPool


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

    def has_devices(self) -> bool:
        """Return whether any devices of this type exist."""
        return self._get_device_count() > 0

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
        context = mp.get_context("spawn")
        return context.Pool(self._get_device_count())

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
        context = tmp.get_context("spawn")
        return context.Pool(self._get_device_count())

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
    burners_all = (cpu, gpu)
    burners_filtered = [burner for burner in burners_all if burner.has_devices()]

    LOG.info("execution started")
    with contextlib.ExitStack() as stack:
        pools = []
        for burner in burners_filtered:
            pool = stack.enter_context(burner.create_pool())
            pools.append(pool)

        if pools:
            for burner, pool in zip(burners_filtered, pools):
                burner.burn_async(
                    config["computation"][burner.device_type_singular],
                    pool,
                )

            for pool in pools:
                pool.close()
                pool.join()

    LOG.info("execution stopped")
    LOG.info("displaying results")
    results: dict[str, Summary] = {}
    for burner in burners_filtered:
        results.update(burner.get_results())

    for device_name, summary in results.items():
        LOG.info("device %s", device_name)
        LOG.info("%s", summary.to_pretty_str())
    LOG.info("results concluded")

    return results


if __name__ == "__main__":
    LOG.info("program started")
    args = conf.get_args()
    config = conf.load_or_build_config(args.config_file)
    results = burn(config)
    to_csv(args.output_file, results)
    LOG.info("program stopped")
