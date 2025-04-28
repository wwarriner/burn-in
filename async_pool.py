"""Implementation of TorchPool threaded async."""

from __future__ import annotations

import asyncio
import functools
import logging
import threading
from typing import TYPE_CHECKING, Any, Self, TypeVar

if TYPE_CHECKING:
    import concurrent.futures
    from collections.abc import Callable, Iterable, Mapping
    from types import TracebackType

LOG = logging.getLogger("burn")

_T = TypeVar("_T")


class TorchPool:
    """Manages a thread with an async event loop."""

    def __init__(self) -> None:
        """Initialize an instance of TorchPool."""
        self._thread: threading.Thread | None = None
        self._async_loop: asyncio.AbstractEventLoop | None = None
        self._futures: list[concurrent.futures.Future[Any]] = []
        self._closed: bool = False
        self._callback: Callable[..., object] | None = None
        self._error_callback: Callable[[BaseException], object] | None = None

    def __enter__(self) -> Self:
        """Start an async loop in a thread to manage apply_async execution."""
        self._async_loop = asyncio.new_event_loop()
        self._async_loop.set_exception_handler(self._exception_callback)
        self._thread = threading.Thread(target=self._run_loop)
        self._thread.daemon = True
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        """Stop loop and thread."""
        self.close()
        self.join()
        return False

    def close(self) -> None:
        """Prevent additional apply_async."""
        self._closed = True

    def join(self) -> None:
        """Join executing tasks."""
        if self._async_loop and self._async_loop.is_running():

            async def _wait_for_tasks() -> None:
                running_tasks = asyncio.all_tasks(self._async_loop)
                other_tasks = [
                    task for task in running_tasks if task is not asyncio.current_task()
                ]
                results = await asyncio.gather(*other_tasks)
                if self._callback is None:
                    raise RuntimeError
                [self._callback(result) for result in results]

            asyncio.run_coroutine_threadsafe(
                _wait_for_tasks(),
                self._async_loop,
            ).result()
            self._async_loop.call_soon_threadsafe(self._async_loop.stop)
        if self._thread:
            self._thread.join(timeout=5)
        if self._async_loop and not self._async_loop.is_closed():
            self._async_loop.close()

        self._thread = None
        self._async_loop = None
        self._futures.clear()
        self._closed = False
        self._callback = None
        self._error_callback = None

    def apply_async(
        self,
        func: Callable[..., _T],
        args: Iterable[Any] = (),
        kwds: Mapping[str, Any] = {},
        callback: Callable[[_T], object] | None = None,
        error_callback: Callable[[BaseException], object] | None = None,
    ) -> None:
        """Begin execution of task."""
        if self._closed:
            return

        if self._async_loop is None or self._thread is None:
            raise RuntimeError

        async_func = asyncio.to_thread(functools.partial(func, *args, **kwds))
        future = asyncio.run_coroutine_threadsafe(async_func, self._async_loop)

        self._callback = callback
        self._error_callback = error_callback

        self._futures.append(future)

    def _run_loop(self) -> None:
        if self._async_loop is None:
            raise RuntimeError

        self._async_loop.run_forever()

    def _exception_callback(
        self,
        loop: asyncio.AbstractEventLoop,  # noqa: ARG002
        context: dict,
    ) -> None:
        if self._error_callback is None:
            return

        self._error_callback(context["exception"])
