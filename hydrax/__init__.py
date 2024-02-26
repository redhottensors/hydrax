"""The main Hydrax module, providing :class:`Dataloader` and :class:`DataGroup`."""

from ._trackedbuffer import TrackedBuffer

import traceback
import sys
import time
import queue
import gc
import os
import warnings

from signal import signal, SIGINT
from types import MappingProxyType
from threading import Thread
from random import Random

import multiprocessing
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import jax # type: ignore[import-not-found]
import jax.numpy as jnp # type: ignore[import-not-found]

from typing import Dict, List, Tuple, Iterable, Callable, Self, Generic, Any, TypeVar

D = TypeVar('D')

def is_worker() -> bool:
    """Returns ``True`` if running in a subprocess.

    Subprocesses cannot instantiate :class:`Dataloader`.
    """

    return multiprocessing.parent_process() is not None

def _as_seq(obj: Any) -> Any:
    if isinstance(obj, dict):
        return list(obj.items())

    if isinstance(obj, str):
        raise TypeError(f"str is not an acceptable sequence type")

    if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        return obj

    if hasattr(obj, "__iter__"):
        return list(obj)

    raise TypeError(f"{type(obj).__name__} is not a sequence or iterable")

class _BatchMemory(TrackedBuffer):
    def __new__(cls, dl: "Dataloader", shm: SharedMemory):
        return super().__new__(cls, shm.buf)

    def __init__(self, dl: "Dataloader", shm: SharedMemory):
        self.dl = dl
        self.shm: SharedMemory | None = shm
        self._rc = 0

    def _ref(self) -> None:
        if self.shm is None:
            raise Exception("batch memory has been recycled")

        self._rc += 1

    def _deref(self) -> None:
        self._rc -= 1
        if self._rc <= 0:
            self.recycle()

    @property
    def name(self) -> str:
        return self.shm.name # type: ignore[union-attr]

    def recycle(self) -> None:
        assert(self._rc == 0)
        assert(self.shm is not None)

        self.dl._recycle_batch(self.shm)
        self.shm = None

class DataGroup(Generic[D]):
    """Represents a group of data which share the same descriptor, batch size, and array shapes.

    :param batch_size: The batch size for loading and processing the data.
    :param arrays: Shape and datatype definitions for all arrays which shall be provided to the loader. Do not include the leading batch dimension.
    :param data: A list of all data descriptors for this group. Descriptors are passed to the loader to identify the data item to load.
        Any finite sequence-like object or iterator is acceptable. The elements must pickleable, as they are sent directly to loader processes.
    """

    def __init__(self, batch_size: int, arrays: Dict[str, Tuple[Tuple[int, ...], np.dtype]], data: Iterable[D]):
        self._data = _as_seq(data)
        self._batch_size = min(len(self._data), batch_size)

        if self._batch_size < 1:
            raise ValueError("batch size must be at least 1")

        self._allocsz = 0

        self._shapes: Dict[str, Tuple[Tuple[int, ...], np.dtype, int, int]] = { }

        for (key, (shape, dtype)) in arrays.items():
            dtype = np.dtype(dtype)
            shape = (self._batch_size, *shape)

            self._allocsz += (-self._allocsz) % dtype.alignment
            count = np.prod(shape).item()

            if count < 1:
                raise ValueError(f"invalid shape for array '{key}': {shape}")

            self._shapes[key] = (shape, dtype, self._allocsz, count)
            self._allocsz += count * dtype.itemsize

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> D:
        return self._data[idx]

    @property
    def batch_size(self) -> int:
        """The batch size configured for the DataGroup."""
        return self._batch_size

    @property
    def batches(self) -> int:
        """The total number of batches this DataGroup can produce per epoch.

        A partial batch cannot be produced, so this is ``len(group) // group.batch_size``.
        """
        return len(self._data) // self._batch_size

    @property
    def arrays(self) -> MappingProxyType[str, Tuple[Tuple[int, ...], np.dtype, int, int]]:
        """Returns the shape and dtype definitions for all arrays provided to the loader. Includes the leading batch dimension."""
        return  MappingProxyType(self._shapes)

    @property
    def memory_size(self) -> int:
        """Returns the allocation size, in bytes, of a single batch."""
        return self._allocsz

    def split(
        self,
        at: int,
        rebatch_first: int | None = None,
        rebatch_second: int | None = None
    ) -> Tuple["DataGroup[D]", "DataGroup[D]"]:
        """Splits a DataGroup into two parts, optionally with different batch sizes.

        The first part contains ``at`` elements (maximum index ``at - 1``), and the second part contains the rest.

        This function is useful to split a dataset into training and validation sets.

        :param at: Index to split the data at, see above.
        :param rebatch_first: New batch size for the first group. If not specified, the current batch size is retained.
        :param rebatch_second: New batch size for the second group. If not specified, the current batch size is retained.
        :return: A tuple consisting of the first group and the second group.
        """

        if rebatch_first is None:
            rebatch_first = self.batch_size

        if rebatch_second is None:
            rebatch_second = self.batch_size

        shapes = {key: (shape[1:], dtype) for (key, (shape, dtype, _, _)) in self._shapes.items()}

        return (
            DataGroup(rebatch_first, shapes, self._data[:at]),
            DataGroup(rebatch_second, shapes, self._data[at:]),
        )

    def _add_arrays(self, batch: Dict[str, Any], memory: _BatchMemory) -> None:
        for (key, (shape, dtype, offset, count)) in self._shapes.items():
            batch[key] = jnp.reshape(jnp.frombuffer(memory, dtype, count, offset), shape)

class _GroupState:
    def __init__(self, data: DataGroup[D], groupid: int, validation: bool):
        self.data = data
        self.groupid = groupid
        self.validation = validation

        self._shuffled = False
        self._indices = [index for index in range(len(data))]
        self._cursor = 0
        self._used = 0.0

    @property
    def used(self) -> float:
        return self._used

    def reset(self) -> None:
        self._cursor = 0
        self._used = 0.0

        if self._shuffled:
            for idx in range(len(self._indices)):
                self._indices[idx] = idx

            self._shuffled = False

    def shuffle(self, rng: Random) -> None:
        self._cursor = 0
        self._used = 0.0
        self._shuffled = True

        rng.shuffle(self._indices)

    def reserve(self) -> List[int] | None:
        end = self._cursor + self.data.batch_size
        if end > len(self.data):
            return None

        indices = self._indices[self._cursor:end]
        self._cursor = end

        if end + self.data.batch_size > len(self.data):
            self._used = 1.0
        else:
            self._used = end / self.data.batch_size

        return indices

def _as_groups(groups: Iterable[DataGroup[D]] | DataGroup[D] | None, first_id: int) -> List[_GroupState]:
    if groups is None:
        return []

    if isinstance(groups, DataGroup):
        return [_GroupState(groups, first_id, first_id != 0)]

    return [_GroupState(group, first_id + idx, first_id != 0) for (idx, group) in enumerate(groups)]

class _BatchLoader:
    def __init__(self, dl: "Dataloader", shm: SharedMemory, group: _GroupState, info: Dict[str, Any]):
        self._dl = dl
        self._size = group.data.batch_size

        self.group = group
        self.memory = _BatchMemory(dl, shm)

        self._remaining = self._size
        self._results: List[Dict[str, Any] | None] = [None for _ in range(self._size)]
        self._batch: Dict[str, Any] | None = info

        self._prior_ready = False
        self._next_ready = False
        self._chained: _BatchLoader | None = None

    def start(self, indices: List[int]) -> None:
        assert(len(indices) == self._size)

        seed = self._batch.get("_seed") # type: ignore[union-attr]

        for idx in range(self._size):
            self._dl._load(self, idx, indices[idx], seed + idx if seed is not None else None)

    def ready(self) -> None:
        self._prior_ready = True
        self._check_chain()

    def chain(self, chained: "_BatchLoader") -> None:
        self._chained = chained
        self._check_chain()

    def load_succeeded(self, idx: int, result: Dict[str, Any]) -> None:
        self._results[idx] = result

        self._remaining -= 1
        if self._remaining == 0:
            self._loading_done()
            self._check_chain()

    def load_failed(self, idx: int) -> None:
        if self._batch is not None:
            self._batch = None
            self._next_ready = True
            self._check_chain()

        self._remaining -= 1
        if self._remaining == 0:
            self._loading_done()

    def _check_chain(self) -> None:
        if not self._prior_ready:
            return

        if self._batch is not None and self._remaining == 0:
            self._dl._batches.put(self._batch)
            self._batch = None

        if not self._next_ready:
            return

        if self._chained is not None:
            self._chained.ready()
            self._chained = None

    def _loading_done(self) -> None:
        if self._batch is not None:
            self._next_ready = True

            try:
                for idx in range(self._size):
                    for (key, value) in self._results[idx].items(): # type: ignore[union-attr]
                        if not key in self._batch:
                            self._batch[key] = [None for _ in range(self._size)]

                        self._batch[key][idx] = value

                self.group.data._add_arrays(self._batch, self.memory)
            except:
                self.memory.recycle()
                self._batch = None
                traceback.print_exc()
        else:
            self.memory.recycle()

        for i in range(self._size):
            self._results[i] = None

class Dataloader:
    """A zero-copy multiprocess JAX dataloader.

    :param loader_func: A callable which accepts a :class:`DataGroup` data descriptor, dictionary of arrays
        (as specified by the DataGroup) corresponding to a single batch element, and an integer seed for use in
        random augmentation and returns a (possibly empty) dictionary of additional data. This callable is called
        repeatedly by loader processes in order to load data items. This callable cannot be a lambda, as it must be
        loadable from a child process. This function must fully populate the passed-in arrays (which may contain
        invalid data from a previous batch), and must not retain any reference to them after completing. The seed
        argument will be ``None`` in the case of validation batches, as validation should operate consistently across
        epochs. The dictionary of additional data must be pickleable, as it is returned to the main process. Do not
        return any of the input arrays; they're already shared with the main process for zero-copy operation. Avoid
        sending any additional arrays via the return dictionary. Instead, add additional zero-copy arrays to the
        DataGroups and fill them in. If for some reason an element cannot be loaded, you must raise an exception or
        allow one to propagate, which will eventually result in the corresponding batch being dropped.
    :param training: DataGroups for training. A single pass through all training DataGroups constitutes an epoch.
    :param validation: An optional tuple specifying a validation mode, interval, and data. The validation mode can
        be either ``"batch"`` or ``"epoch"``, and the interval is how many batches or epochs between validation runs.
    :param loader_depth: The maximum number of batches that can exist at any point in time. Memory usage is
        proportional to the size of the largest possible batch multiplied by the loader depth. This should be at least
        two (one batch being processed, one batch loading), but should be larger if the dataloader needs to work ahead
        further to amortize loading time outliers. The default is 3, allowing the dataloader to work one batch ahead.
    :param loader_count: The number of loader processes. Each loader process loads a single item at a time. This
        defaults to 1, but should probably be higher. Optimally, it should be tuned to saturate the available
        throughput of your data origin (disk/network) without introducing unnecessary process context switching.
    :param start_at: A tuple specifying how far to skip ahead before loading, for example to resume from a checkpoint.
        The first element is the epoch to skip to, and the second is a number of additional batches to skip. The number
        of batches to skip can exceed the number of batches in an epoch, in which case additional epochs are skipped.
        The default is ``(0, 0)``, indicating to start at the beginning.
    :param end_at: An optional tuple specifying when to stop. The first element is either ``"batch"`` or ``"epoch"``,
        and the second specifies which zero-indexed batch or epoch to stop before. So ``("epoch", 1)`` stops after one
        whole epoch. This argument specifies an absolute position and is not relative to ``start_at``. If this is not
        specified, the dataloader runs until it is interrupted by either :func:`interrupt` or ``KeyboardInterrupt``.
    :param interleave_groups: If multiple training DataGroups are specified, this controls how batches from the
        different groups are interleaved within an epoch. If ``False``, groups are loaded sequentially in the order
        specified. If ``True``, the default, batches from different groups are interleaved, with the least-utilized
        earliest-index group being selected for each batch.
    :param shuffle_groups: Specifies how to shuffle data between epochs. The default, ``"later"``, randomly (but
        deterministically) permutes all epochs except the first. ``"never"`` runs all groups in-order, and ``"all"``
        permutes all epochs including the first.
    :param seed: Specifies a seed used for randomness. This seed is used to deterministically calculate permutations
        for ``shuffle_groups`` and the augmentation seeds passed to ``loader_func``. The default is ``0``.

    Example::

        from hydrax import Dataloader, DataGroup

        def my_loader(data, arrays, seed):
            # load data from data source into arrays, optionally augmenting using 'seed'.
            # if 'seed' is None this is a validation batch
            # return any additional data for the batch

        if __name__ == "main":
            my_data = ...
            array_defs = {
                "array_name": ((dim, ...), dtype),
                ...
            }

            all_data = DataGroup(batch_size, array_defs, my_data)
            valid, train = all_data.split(1000) # or however many validation items to reserve

            dataloader = Dataloader(
                my_loader,
                train,
                valid = ("epoch", 1, valid), # run validation after every epoch
                end_at = ("epoch", 5)        # run 5 epochs in total
            )

            with dataloader: # a with block is required
                # consider using hydrax.tqdm.tbatches instead of a vanilla for loop here
                for batch in dataloader:
                    if batch['_validation']:
                        run_validation_batch(batch)
                    else:
                        run_training_batch(batch)

    Each batch produced by :func:`__next__` has the following structure::

        {
            "_validation": bool, # True if this is a validation batch, False if this is a training batch
            "_group": int,       # index of the source group, in the order specified to __init__

            # present for training batches only
            "_epoch": int,       # the current epoch, starting at 0
            "_epoch_batch": int, # the current batch within this epoch, starting at 0
            "_batch": int,       # the overall batch number, not including validation batches
            "_seed": int,        # a seed for randomness, unique to this batch

            # present for validation batches only
            "_validation_epoch": int,       # the currrent validation epoch, always starting at 0
            "_validation_epoch_batch": int, # the current batch within this validation epoch, starting at 0
            "_validation_batch": int,       # the overall validation batch number

            # for each array specified by the current data group
            "array_name": jax.Array, # dim = (batch_size, ...)
            ...,

            # for each additional data key returned by the loader
            # if the loader did not return the key for a specific item, its corresponding element is None
            "additional_key": [ value_0, ... ], # len = batch_size
            ...,
        }
    """

    def __init__(
        self,
        loader_func: Callable[[D, MappingProxyType[str, np.ndarray], int | None], Dict[str, Any]],
        training: Iterable[DataGroup[D]] | DataGroup[D],
        validation: Tuple[str, int, Iterable[DataGroup[D]] | DataGroup[D]] | None = None, # ("epoch" | "batch"), interval, data
        loader_depth: int = 3,
        loader_count: int = 1,
        start_at: Tuple[int, int] = (0, 0), # epoch, epoch batch
        end_at: Tuple[str, int] | None = None, # ("never" | "epoch" | "batch"), interval
        interleave_groups: bool = True,
        shuffle_groups: str = "later", # ("none" | "later" | "all")
        seed: int = 0,
    ):
        if is_worker():
            raise Exception("Dataloader cannot run in a worker process.")

        if shuffle_groups not in ("none", "later", "all"):
            raise ValueError("shuffle_groups must be 'none', 'later', or 'all'.")

        self._loader = loader_func
        self._start_at = start_at
        self._end_at = end_at if end_at is not None else ("never", -1)
        self._interleave = interleave_groups
        self._shuffle = shuffle_groups
        self._seed = seed

        if self._end_at[0] not in ('never', 'epoch', 'batch'):
            raise ValueError("end_at endpoint must be 'never', 'epoch', or 'batch'")

        self._tgroups = _as_groups(training, 0)
        self._memsz = max(group.data.memory_size for group in self._tgroups)
        self._maxbatch = max(group.data.batch_size for group in self._tgroups)

        if len(self._tgroups) == 1:
            self._interleave = False

        if validation is not None:
            if not validation[0] in ("epoch", "batch"):
                raise ValueError("validation mode must be 'epoch' or 'batch'")

            self._vmode = validation[0]
            self._vinterval = validation[1]
            self._vgroups = _as_groups(validation[2], len(self._tgroups))
            self._vepoch = -1
            self._vbatch = -1

            self._memsz = max(self._memsz, max(group.data.memory_size for group in self._vgroups))
            self._batches_per_validation = sum(group.data.batches for group in self._vgroups)
        else:
            self._vmode = "none"
            self._batches_per_validation = 0

        self._setup = False
        self._running = False
        self._interrupt = False
        self._validating = False

        self._idle_usec = 0
        self._first_batch = True

        if not interleave_groups:
            self._tgroup = -1

        self._mpctx = multiprocessing.get_context("spawn")
        self._batches = queue.SimpleQueue[Dict[str, Any] | None]()
        self._buffers = queue.SimpleQueue[SharedMemory]()
        self._memories: List[SharedMemory | None] = [None for _ in range(loader_depth)]

        self._submission_id = 0
        self._submission_queue = self._mpctx.SimpleQueue() # type: multiprocessing.SimpleQueue[Tuple[int, D, str, int, int, int | None] | None]
        self._completion_queue = self._mpctx.SimpleQueue() # type: multiprocessing.SimpleQueue[Tuple[int, Dict[str, Any] | None] | None]
        self._inflight: Dict[int, Tuple[_BatchLoader, int]] = {}

        self._submission_thread: Thread | None = None
        self._completion_thread: Thread | None = None
        self._loaders: List[Process | None] = [None for _ in range(loader_count)]

        self._chain_to: _BatchLoader | None = None
        self._batches_per_epoch = sum(group.data.batches for group in self._tgroups)
        self._sigint: Any = None

        while self._start_at[1] >= self._batches_per_epoch:
            self._start_at = (self._start_at[0] + 1, self._start_at[1] - self._batches_per_epoch)

    def __enter__(self) -> Self:
        assert(not self._setup)

        self._setup = True
        self._running = True
        self._validating = False
        self._interrupt = False

        self._sigint = signal(SIGINT, self._handle_sigint)
        self._idle_usec = 0
        self._first_batch = True

        if not self._interleave:
            self._tgroup = 0

        if self._vmode != "none":
            self._vepoch = 0
            self._vbatch = 0

        for i in range(len(self._memories)):
            memory = SharedMemory(create=True, size=self._memsz)
            self._memories[i] = memory
            self._buffers.put(memory)

        tshapes = [group.data._shapes for group in self._tgroups]
        vshapes = [group.data._shapes for group in self._vgroups] if self._vmode != "none" else []

        loader_args = (
            self._loader,
            self._submission_queue,
            self._completion_queue,
            [memory.name for memory in self._memories], # type: ignore[union-attr]
            tshapes + vshapes
        )

        for i in range(len(self._loaders)):
            loader = self._mpctx.Process(
                target=_run_loader,
                name=f"hydrax-loader-{i}",
                args=loader_args,
                daemon=True
            )
            loader.start()

            self._loaders[i] = loader # type: ignore

        self._completion_thread = Thread(
            target=self._run_completion,
            name="hydrax-completion",
            daemon=True
        )
        self._completion_thread.start()

        self._submission_thread = Thread(
            target=self._run_submission,
            name="hydrax-submission",
            daemon=True
        )
        self._submission_thread.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        assert(self._setup)

        if self._running:
            self._interrupt = True
            self.__next__()

        signal(SIGINT, self._sigint)
        self._sigint = None

        if self._submission_thread is not None:
            self._submission_thread.join()
            self._submission_thread = None

        gc.collect()

        for i in range(len(self._memories)):
            memory = self._memories[i]

            if memory is not None:
                memory.close()
                memory.unlink()
                self._memories[i] = None

        self._setup = False

    def __iter__(self) -> Self:
        assert(self._setup)

        return self

    def __next__(self) -> Dict[str, Any]:
        assert(self._running)

        if self._interrupt:
            while self._batches.get() is not None:
                pass

            self._running = False
            raise StopIteration

        try:
            batch = self._batches.get_nowait()
        except queue.Empty:
            start = time.monotonic_ns()
            batch = self._batches.get()
            end = time.monotonic_ns()

            if batch is not None and not self._first_batch:
                self._idle_usec += (end - start) // 1000

        self._first_batch = False

        if batch is None:
            self._running = False
            raise StopIteration

        return batch

    @property
    def batches_per_epoch(self) -> int:
        """The total number of batches per epoch."""

        return self._batches_per_epoch

    @property
    def batches_per_validation(self) -> int:
        """The total number of batches per validation run.

        If no validation data was specified, this is ``0``.
        """

        return self._batches_per_validation

    @property
    def first_batch(self) -> int:
        """Returns the index of the first batch to load. Controlled by the ``start_at`` argument."""
        return self._start_at[0] * self._batches_per_epoch + self._start_at[1]

    @property
    def last_batch(self) -> int | None:
        """Returns the index of the final batch to load. Controlled by the ``end_at`` argument.

        Will be ``None`` if no end point was specified.
        """
        match self._end_at[0]:
            case "batch":
                return self._end_at[1]
            case "epoch":
                return self._end_at[1] * self._batches_per_epoch
            case _:
                return None

    def idle_usec(self) -> int:
        """Returns the total amount of time, in microseconds, since the last call to ``idle_usec``, that the
        :func:`__next__` has spent waiting for a batch.

        This represents the amount of time that the dataloader has stalled JAX dispatch. Ideally, this value should
        always be zero. If it is consistently high, you either have too few loaders (``loader_count``) or are
        bottlenecked by a shared resource (disk / network / cpu / swap). If it has spikes, you may need to increase
        ``loader_depth`` to allow loaders to work ahead in order to amortize longer loading times.

        :func:`hydrax.tqdm.tbatches` consumes this metric if ``report_interval`` is specified. Do not use it if you
        are using ``tbatches`` for stall reporting.
        """

        value = self._idle_usec
        self._idle_usec = 0
        return value

    def interrupt(self):
        """Interrupts the dataloader, so no further batches are returned by :func:`__next__`."""

        assert(self._setup)
        self._interrupt = True

    def _load(self, loader: _BatchLoader, batch_idx: int, data_idx: int, seed: int | None) -> None:
        sid = self._submission_id
        self._submission_id += 1

        self._inflight[sid] = (loader, batch_idx)

        try:
            self._submission_queue.put((
                sid,
                loader.group.data[data_idx],
                loader.memory.name,
                loader.group.groupid,
                batch_idx,
                seed
            ))
        except:
            traceback.print_exc()
            loader.load_failed(batch_idx)

    def _run_submission(self) -> None:
        remaining = len(self._memories)

        if not self._submit_batches():
            remaining -= 1

        self._chain_to = None

        for _ in range(len(self._loaders)):
            self._submission_queue.put(None)

        for idx in range(len(self._loaders)):
            self._loaders[idx].join() # type: ignore[union-attr]
            self._loaders[idx] = None

        self._completion_queue.put(None)
        self._completion_thread.join() # type: ignore[union-attr]
        self._completion_thread = None

        for _ in range(remaining):
            self._buffers.get()

        self._batches.put(None)

    def _submit_batches(self) -> bool:
        (epoch, seek) = self._start_at
        batch = epoch * self._batches_per_epoch

        while True:
            self._init_groups(epoch)
            ebatch = 0

            while (group := self._select_group()) is not None:
                while (indices := group.reserve()) is not None:
                    if seek == 0:
                        if not self._chain_load(group, indices, {
                            "_validation": False,
                            "_group": group.groupid,
                            "_epoch": epoch,
                            "_epoch_batch": ebatch,
                            "_batch": batch,
                            "_seed": self._seed + (batch * self._maxbatch),
                        }):
                            return False
                    else:
                        seek -= 1

                    ebatch += 1
                    batch += 1

                    if seek == 0 and self._vmode == "batch" and batch % self._vinterval == 0:
                        if not self._submit_validation():
                            return False

                    if self._end_at[0] == "batch" and batch == self._end_at[1]:
                        return True

            epoch += 1

            if seek == 0 and self._vmode == "epoch" and epoch % self._vinterval == 0:
                if not self._submit_validation():
                    return False

            if self._end_at[0] == "epoch" and epoch == self._end_at[1]:
                return True

    def _init_groups(self, epoch: int) -> None:
        if self._shuffle == "none" or (self._shuffle == "later" and epoch == 0):
            for group in self._tgroups:
                group.reset()
        else:
            rng = Random(self._seed + epoch)
            for group in self._tgroups:
                group.shuffle(rng)

    def _select_group(self) -> _GroupState | None:
        if self._interleave:
            group = min(self._tgroups, key=lambda group: group.used)
            return group if group.used < 1.0 else None
        else:
            idx = self._tgroup
            self._tgroup += 1

            if idx >= len(self._tgroups):
                self._tgroup = 0
                return None

            return self._tgroups[idx]

    def _submit_validation(self) -> bool:
        vebatch = 0

        for vgidx in range(len(self._vgroups)):
            vgroup = self._vgroups[vgidx]

            while (indices := vgroup.reserve()) is not None:
                if not self._chain_load(vgroup, indices, {
                    "_validation": True,
                    "_group": vgidx,
                    "_validation_epoch": self._vepoch,
                    "_validation_epoch_batch": vebatch,
                    "_validation_batch": self._vbatch,
                }):
                    return False

                vebatch += 1
                self._vbatch += 1

        self._vepoch += 1
        return True

    def _chain_load(self, group: _GroupState, indices: List[int], info: Dict[str, Any]) -> bool:
        shm = self._buffers.get()
        if self._interrupt:
            return False

        loader = _BatchLoader(self, shm, group, info)
        loader.start(indices)

        if self._chain_to is None:
            loader.ready()
        else:
            self._chain_to.chain(loader)

        self._chain_to = loader
        return True

    def _run_completion(self) -> None:
        while (completion := self._completion_queue.get()) is not None:
            (sid, result) = completion
            (loader, batch_idx) = self._inflight.pop(sid)

            if result is None:
                loader.load_failed(batch_idx)
            else:
                loader.load_succeeded(batch_idx, result)

    def _recycle_batch(self, shm: SharedMemory) -> None:
        self._buffers.put(shm)

    def _handle_sigint(self, signum, frame) -> None:
        if not self._interrupt:
            print("[hydrax] KeyboardInterrupt")
            self.interrupt()

def _run_loader(
    loader: Callable[[D, MappingProxyType[str, np.ndarray], int | None], Dict[str, Any]],
    submission_queue: "multiprocessing.SimpleQueue[Tuple[int, D, str, int, int, int | None] | None]",
    completion_queue: "multiprocessing.SimpleQueue[Tuple[int, Dict[str, Any] | None] | None]",
    memory_names: List[str],
    group_shapes: List[Dict[str, Tuple[Tuple[int, ...], np.dtype, int, int]]],
) -> None:
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    memories: Dict[str, SharedMemory] = {}

    try:
        for memory_name in memory_names:
            memories[memory_name] = SharedMemory(memory_name)

        while (work := submission_queue.get()) is not None:
            (sid, data, memory_name, group_id, batch_idx, seed) = work

            try:
                shapes = group_shapes[group_id]
                buffer = memories[memory_name].buf

                result = loader(data, MappingProxyType({
                    key: np.frombuffer(buffer, dtype, count, offset).reshape(shape)[batch_idx]
                    for (key, (shape, dtype, offset, count)) in shapes.items()
                }), seed)

                if not isinstance(result, dict):
                    raise ValueError(f"loader produced a non-dict result of type: {type(result).__name__}")

                for key, value in result.items():
                    if key.startswith("_"):
                        raise ValueError(f"loader produced reserved key: '{key}'")

                    if key in shapes:
                        raise ValueError(f"loader produced conflicting key: '{key}'")

                    if isinstance(value, np.ndarray) or isinstance(value, jax.Array):
                        warnings.warn(f"loader produced an array type with key: '{key}' ({type(value).__name__})", RuntimeWarning)

                completion_queue.put((sid, result))

                del result
            except:
                completion_queue.put((sid, None))

                print(f"[hydrax] exception while loading data item: {data}")
                traceback.print_exc()
    finally:
        # these may be holding on to a shm if the user is misbehaving
        try:
            del result
        except UnboundLocalError:
            pass

        try:
            del value
        except UnboundLocalError:
            pass

        gc.collect()

        for memory in memories.values():
            memory.close()
