"""The main Hydrax module, providing :class:`Dataloader`, :class:`DataGroup`, and :class:`Batch`."""

from ._trackedbuffer import TrackedBuffer

import traceback
import time
import queue
import gc
import os

from signal import signal, SIGINT
from threading import Thread
from types import MappingProxyType
from warnings import warn

import multiprocessing
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from numpy.random import Generator, PCG64

import jax # type: ignore[import-not-found]
import jax.numpy as jnp # type: ignore[import-not-found]

from typing import Dict, List, Tuple, Iterable, Sequence, Callable, Self, Generic, Any, TypeVar

class BatchTimeoutError(TimeoutError):
    """Raised if no batch has been produced for some time.

    Controlled by the ``timeout_sec`` argument to :class:`Dataloader`.
    """
    pass

def is_worker() -> bool:
    """Returns ``True`` if running in a subprocess.

    Subprocesses cannot instantiate :class:`Dataloader`.
    """

    return multiprocessing.parent_process() is not None

def _as_seq(obj: Any) -> Any:
    if isinstance(obj, dict) or isinstance(obj, MappingProxyType):
        return list(obj.items())

    if isinstance(obj, str):
        return [obj]

    if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        return obj

    if hasattr(obj, "__iter__"):
        return list(obj)

    raise TypeError(f"{type(obj).__name__} is not a sequence or iterable")

class _BatchMemory(TrackedBuffer):
    __slots__ = ("shm", "_queue", "_rc")

    def __new__(cls, dataloader: "Dataloader", shm: SharedMemory):
        return super().__new__(cls , shm.buf) # type: ignore[arg-type]

    def __init__(self, dataloader: "Dataloader", shm: SharedMemory):
        self.shm: SharedMemory | None = shm
        self._queue = dataloader._buffers
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

        self._queue.put(self.shm)
        self.shm = None

class _LoaderMemory(TrackedBuffer):
    __slots__ = ("_rc", "_frozen")

    def __new__(cls, shm: SharedMemory):
        return super().__new__(cls, shm.buf) # type: ignore[arg-type]

    def __init__(self, buf: Any):
        self._rc = 0
        self._frozen = False

    def _ref(self) -> None:
        if self._frozen:
            raise Exception(f"loader memory has been transferred")

        self._rc += 1

    def _deref(self) -> None:
        self._rc -= 1

    def freeze(self) -> None:
        self._frozen = True

        if self._rc > 0:
            warn("loader memory has dangling reference", RuntimeWarning)

D = TypeVar('D')
class DataGroup(Generic[D]):
    """Represents a group of data which share the same descriptor, batch size, and array shapes.

    .. caution::
        Don't derive from DataGroup, and do not modify your dataset after placing it in a DataGroup.

    :param batch_size: The batch size for loading and processing the data.
    :param arrays: Shape and datatype definitions for all arrays in a batch. Do not include the leading batch
        dimension, since that is specified by ``batch_size``. These arrays will be presented to the loader for
        zero-copy initialization.
    :param data: A list of all data descriptors for this group. Descriptors are passed to the loader to identify
        the data item to load. Any finite sequence-like object or iterator is acceptable. The elements must
        pickleable, as they are sent directly to loader processes.

    .. tip::
        If you want to split a dataset into traning and validation batches, use :func:`split`.

        If you want to repeat a DataGroup multiple times per epoch, use :func:`clone`. This avoids creating
        multiple copies of your dataset and allows the data to be shuffled independently.

    .. warning::
        If your dataset is an iterable and not otherwise indexable, it will be materialized by the DataGroup.
        If you have hundreds of thousands of items, consider using the :mod:`hydrax.pandas` adapter module.
    """

    __slots__ = ("_data", "_batch_size", "_allocsz", "_shapes")

    def __init__(
        self,
        batch_size: int,
        arrays: Dict[str, Tuple[Tuple[int, ...], np.dtype]],
        data: Iterable[D]
    ):
        if type(self) is not DataGroup:
            warn(f"{type(self).__name__} derives from hydrax.DataGroup. This is not supported.", SyntaxWarning)

        self._data = _as_seq(data)
        self._batch_size = min(len(self._data), batch_size)

        if self._batch_size < 1:
            raise ValueError("batch size must be at least 1")

        self._allocsz = 0
        self._shapes: Dict[str, Tuple[Tuple[int, ...], np.dtype, int, int]] = { }

        for (key, (shape, dtype)) in arrays.items():
            dtype = np.dtype(dtype)
            shape = (self._batch_size, *shape)
            count = np.prod(shape).item()

            if count < 1:
                raise ValueError(f"invalid shape for array '{key}': {shape}")

            self._allocsz += (-self._allocsz) % dtype.alignment
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
        """The shape and dtype definitions for all arrays provided to the loader. Includes the leading batch
        dimension.

        The mapping has the form: ``{ 'array_name': ((batch_size, dim_1, ...), dtype, offset, count), ... }``
        """

        return MappingProxyType(self._shapes)

    @property
    def memory_size(self) -> int:
        """The allocation size, in bytes, of a single batch."""
        return self._allocsz

    def split(
        self,
        at: int,
        rebatch_first: int | None = None,
        rebatch_second: int | None = None
    ) -> Tuple["DataGroup[D]", "DataGroup[D]"]:
        """Splits a DataGroup into two parts, optionally with different batch sizes.

        :param at: Index to split the data at, see above.
        :param rebatch_first: New batch size for the first group. If not specified, the current batch size is retained.
        :param rebatch_second: New batch size for the second group. If not specified, the current batch size is
            retained.
        :return: A tuple consisting of the first group and the second group.

        The first part contains ``at`` elements (maximum index ``at - 1``), and the second part contains the rest.

        This function is useful to split a dataset into training and validation sets.
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

    def clone(self, rebatch: int | None = None) -> "DataGroup[D]":
        """Creates an identical copy of the DataGroup, referencing the same data.

        :param rebatch: The new batch size for the clone. If not specified, the current batch size is retained.

        This function is useful to repeat a given group of data multiple times within each epoch. You must arrange
        the clones according to your interleaving preferences and pass them to :func:`Dataloader._init_`.
        """

        if rebatch is None:
            rebatch = self._batch_size

        shapes = {key: (shape[1:], dtype) for (key, (shape, dtype, _, _)) in self._shapes.items()}

        return DataGroup(rebatch, shapes, self._data)

    def _add_arrays(self, batch: Dict[str, Any], memory: _BatchMemory) -> None:
        for (key, (shape, dtype, offset, count)) in self._shapes.items():
            batch[key] = jnp.reshape(jnp.frombuffer(memory, dtype, count, offset), shape)

class _Items(Sequence):
    __slots__ = ("_array")

    def __init__(self, array: np.ndarray):
        self._array = array

    def __len__(self) -> int:
        return self._array.size

    def __getitem__(self, key):
        return self._array.item(key)

class _GroupState(Generic[D]):
    __slots__ = ("data", "groupid", "_indices", "_cursor", "_used")

    def __init__(self, data: DataGroup[D], groupid: int):
        if not isinstance(data, DataGroup):
            raise TypeError(f"data of type {type(data).__name__} is not in a hydrax.DataGroup")

        self.data = data
        self.groupid = groupid

        self._indices: np.ndarray | None = None
        self._cursor = 0
        self._used = 0.0

    @property
    def used(self) -> float:
        return self._used

    def reset(self) -> None:
        self._cursor = 0
        self._used = 0.0
        self._indices = None

    def shuffle(self, rng: Generator) -> None:
        self._cursor = 0
        self._used = 0.0
        self._indices = rng.permutation(len(self.data))

    def reserve(self) -> Sequence[int] | None:
        dlen = len(self.data)
        end = self._cursor + self.data.batch_size

        if end > dlen:
            return None

        if self._indices is None:
            indices = range(self._cursor, end) # type: Sequence[int]
        else:
            indices = _Items(self._indices[self._cursor:end])

        self._cursor = end

        if end + self.data.batch_size > dlen:
            self._used = 1.0
        else:
            self._used = end / dlen

        return indices

def _as_groups(groups: Iterable[DataGroup[D]] | DataGroup[D] | None, first_id: int) -> List[_GroupState]:
    if groups is None:
        return []

    if isinstance(groups, DataGroup):
        return [_GroupState(groups, first_id)]

    return [_GroupState(group, first_id + idx) for (idx, group) in enumerate(groups)]

class _BatchData(Sequence[D]):
    __slots__ = ("_batch")

    def __init__(self, batch: "Batch[D]"):
        self._batch = batch

    def __len__(self) -> int:
        return len(self._batch._indices)

    def __getitem__(self, key):
        return self._batch._group[self._batch._indices[key]]

class Batch(Generic[D]):
    """An abstract base class representing a batch of data loaded by a :class:`Dataloader`.

    Batches are returned by :func:`Dataloader.__next__`.

    If ``is_training`` is ``True``, this is a :class:`TrainingBatch`. Otherwise, ``is_validation`` is ``True`` and this
    instance is a :class:`ValidationBatch`.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.
    """

    __slots__ = ("_group", "_indices", "_arrays", "_additional")

    def __init__(self, group: DataGroup[D], indices: Sequence[int]):
        self._group = group
        self._indices = indices
        self._arrays: Dict[str, jax.Array] = {}
        self._additional: Dict[str, List[Any]] = {}

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def is_validation(self) -> bool:
        """``True`` if this instance is a :class:`ValidationBatch`, and ``False`` otherwise."""

        raise NotImplemented

    @property
    def is_training(self) -> bool:
        """``True`` if this instance is a :class:`TrainingBatch`, and ``False`` otherwise."""

        raise NotImplemented

    @property
    def group(self) -> DataGroup[D]:
        """The DataGroup from which this batch was loaded."""

        return self._group

    @property
    def indices(self) -> Sequence[int]:
        """The corresponding indices in ``group`` of the data in this batch."""

        return self._indices

    @property
    def data(self) -> Sequence[D]:
        """The data descriptors of the data in this batch."""

        return _BatchData(self)

    def get_data(self, index: int) -> D:
        """Returns the data descriptor for the specified item in this batch."""

        return self._group[self._indices[index]]

    @property
    def arrays(self) -> MappingProxyType[str, jax.Array]:
        """The JAX arrays for this batch, as defined by the :class:`DataGroup`.

        The leading dimension of each array is the batch size of the ``DataGroup``. The shapes and dtypes are as
        specified by ``batch.group.arrays``.
        """

        return MappingProxyType(self._arrays)

    def get_array(self, name: str) -> jax.Array:
        """Returns the specified JAX array for this batch, as defined in the :class:`DataGroup`.

        The leading dimension of the returned array is the batch size of the ``DataGroup``. The shape and dtype are as
        specified by ``batch.group.arrays[name]``.

        This is equivalent to ``batch.arrays[name]``.

        :param name: The name of the array, as defined in the `DataGroup`.
        """

        return self._arrays[name]

    @property
    def additional(self) -> MappingProxyType[str, List[Any]]:
        """The additional data returned by the loader for each item of the batch.

        This data is a readonly mapping of the form ``{ 'key_0': [value_0, ...], ... }``, where the number of values
        for each key is equal to the batch size of the ``DataGroup``. A value will be ``None`` if the loader did not
        return additional data with the corresponding key for the corresponding item in this batch.

        .. caution:
            If your loader function does not return a given key for all data items that it loads, it is possible you
            may encounter a batch where no data item produces the specified key, and so it is not present in the
            returned mapping. If this is the case, consider using :func:`get_additional` instead.
        """

        return MappingProxyType(self._additional)

    def get_additional(self, key: str, index: int, default: Any = None) -> Any:
        """Returns the additional data returned by the loader with the specified name for the specified item.

        The result will be ``default`` (``None``, unless otherwise specified) if the loader did not return additional
        data with the corresponding key for the specified item, or if such additional data was itself ``None``.

        :param key: The key returned by the loader.
        :param index: The index of the data item within this batch.
        :param default: The default value to return if the additional data is not present for the item.
        :return: The additional data value returned by the loader, or ``default``.
        """

        additional = self._additional.get(key)
        if additional is None:
            if index < -len(self._indices) or index >= len(self._indices):
                raise IndexError

            return default

        value = additional[index]
        if value is None:
            return default

        return value

    def _load(self, bl: "_BatchLoader[D]") -> None:
        raise NotImplemented

    def _item_loaded(self, idx: int, additional: Dict[str, Any]) -> None:
        for (key, value) in additional.items():
            if not key in self._additional:
                self._additional[key] = [None for _ in range(len(self))]

            self._additional[key][idx] = value

    def _arrays_loaded(self, memory: _BatchMemory) -> None:
        self._group._add_arrays(self._arrays, memory)

class TrainingBatch(Batch[D], Generic[D]):
    """A batch of training data loaded by a :class:`Dataloader`.

    Batches are returned by :func:`Dataloader.__next__`. You can determine if a :class:`Batch` is a ``TrainingBatch``
    by checking ``is_training``.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.
    """

    __slots__ = ("_epoch", "_epoch_batch", "_batch", "_seed", "_seeds")

    def __init__(
        self,
        group: DataGroup[D],
        indices: Sequence[int],
        epoch: int,
        epoch_batch: int,
        batch: int,
        seed: int,
    ):
        super().__init__(group, indices)

        self._epoch = epoch
        self._epoch_batch = epoch_batch
        self._batch = batch
        self._seed = seed
        self._seeds = range(seed, seed + len(indices))

    @property
    def is_validation(self) -> bool:
        """``False``"""

        return False

    @property
    def is_training(self) -> bool:
        """``True``"""

        return True

    @property
    def epoch(self) -> int:
        """The zero-based training epoch number for this batch."""

        return self._epoch

    @property
    def epoch_batch(self) -> int:
        """The zero-based index of this batch within the current epoch."""

        return self._epoch_batch

    @property
    def batch_num(self) -> int:
        """The overall zero-based index of this batch."""

        return self._batch

    @property
    def seed(self) -> int:
        """The deterministic seed for randomness associated with this batch."""

        return self._seed

    @property
    def seeds(self) -> Sequence[int]:
        """The deterministic seeds for randomness associated with each item of this batch.

        Each seed is the same seed that was passed to the :class:`Dataloader` ``loader_func`` for the corresponding
        item of this batch.
        """

        return self._seeds

    def _load(self, bl: "_BatchLoader[D]") -> None:
        for (batch_idx, data_idx) in enumerate(self._indices):
            bl.dataloader._load(bl, batch_idx, data_idx, self._seeds[batch_idx])

class ValidationBatch(Batch[D], Generic[D]):
    """A batch of validation data loaded by a :class:`Dataloader`.

    Batches are returned by :func:`Dataloader.__next__`. You can determine if a :class:`Batch` is a ``ValidationBatch``
    by checking ``is_validation``.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.
    """

    __slots__ = ("_epoch", "_epoch_batch", "_batch")

    def __init__(
        self,
        group: DataGroup[D],
        indices: Sequence[int],
        epoch: int,
        epoch_batch: int,
        batch: int,
    ):
        super().__init__(group, indices)
        self._epoch = epoch
        self._epoch_batch = epoch_batch
        self._batch = batch

    @property
    def is_validation(self) -> bool:
        """``True``"""

        return True

    @property
    def is_training(self) -> bool:
        """``False``"""

        return False

    @property
    def validation_epoch(self) -> int:
        """The zero-based validation epoch number for this batch.

        Unlike a training epoch number, this number starts at zero regardless of how many validation epochs were
        skipped by the ``start_at`` argument to :class:`Dataloader`.
        """

        return self._epoch

    @property
    def validation_epoch_batch(self) -> int:
        """The zero-based index of this batch within the current validation epoch."""

        return self._epoch_batch

    @property
    def validation_batch_num(self) -> int:
        """The overall zero-based index of this validation batch.

        Unlike a training batch number, this number starts at zero regardless of how many validation batches were
        skipped by the ``start_at`` argument to :class:`Dataloader`. This number counts separately from training batch
        numbers.
        """

        return self._batch

    def _load(self, bl: "_BatchLoader[D]") -> None:
        for (batch_idx, data_idx) in enumerate(self._indices):
            bl.dataloader._load(bl, batch_idx, data_idx, None)

class _BatchLoader(Generic[D]):
    __slots__ = ("dataloader", "batch", "group", "memory", "_remaining", "_prior_ready", "_next_ready", "_chained")

    def __init__(
        self,
        dataloader: "Dataloader[D]",
        batch: Batch[D],
        group: _GroupState[D],
        shm: SharedMemory
    ):
        self.dataloader = dataloader
        self.batch: Batch[D] | None = batch
        self.group = group
        self.memory = _BatchMemory(dataloader, shm)

        self._remaining = len(batch)
        self._prior_ready = False
        self._next_ready = False
        self._chained: _BatchLoader | None = None

    def ready(self) -> None:
        self._prior_ready = True
        self._check_chain()

    def chain(self, chained: Self) -> None:
        self._chained = chained
        self._check_chain()

    def load_succeeded(self, idx: int, result: Dict[str, Any]) -> None:
        if self.batch is not None:
            self.batch._item_loaded(idx, result)

        self._remaining -= 1
        if self._remaining == 0:
            if self.batch is not None:
                self.batch._arrays_loaded(self.memory)
                self._check_chain()
            else:
                self.memory.recycle()

    def load_failed(self, idx: int) -> None:
        if self.batch is not None:
            self.batch = None
            self._next_ready = True
            self._check_chain()

        self._remaining -= 1
        if self._remaining == 0:
            self.memory.recycle()

    def _check_chain(self) -> None:
        if not self._prior_ready:
            return

        if self._remaining == 0 and self.batch is not None:
            self.dataloader._batches.put(self.batch)
            self.batch = None
            self._next_ready = True
        elif not self._next_ready:
            return

        if self._chained is not None:
            self._chained.ready()
            self._chained = None

class Dataloader(Generic[D]):
    """A zero-copy multiprocess JAX dataloader.

    .. caution::
        Don't derive from ``Dataloader``. Everything customizable is provided as an argument.

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
    :param training: A :class:`DataGroup` for training, or any iterable of them. A single pass through all training
        DataGroups constitutes an epoch.
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
    :param timeout_sec: Raise :class:`BatchTimeoutError` if no batches have completed within the specified timeout.
        The default is ``60``, and ``0`` or less disables.

    .. tip::
        In Hydrax, a single Dataloader is usually responsible for producing both your training and validation batches,
        in order to conserve resources and ensure perfectly smooth loading throughout.

    Example::

        from hydrax import Dataloader, DataGroup, TrainingBatch, ValidationBatch

        def my_loader(data, arrays, seed):
            # load data from data source into arrays, optionally augmenting using 'seed'.
            # if 'seed' is None this is a data from a validation batch
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
                    if isinstance(batch, TrainingBatch):
                        run_training_batch(batch)
                    elif isinstance(batch, ValidationBatch):
                        run_validation_batch(batch)

    .. important::
        Read the documentation for your ``loader_func`` carefully. If you receive a warning from Hydrax about
        your loader, you should fix your code. Failure to do this could result in your batch data changing out
        from underneath you, leading to significant training issues such as NaNs.

    .. warning::
        Do not attempt to construct a Dataloader inside a loader process. Ensure your training code is guarded
        with ``if __name__ == '__main__':``, or is otherwise prevented from running. As a last resort, you can
        check :func:`hydrax.is_worker` and bail.

    .. note::
        The Dataloader installs a handler for ``KeyboardInterrupt`` (Ctrl+C / SIGINT), which stops the flow of
        batches as soon as possible. After the dataloader has completed, you can check if this occurred by
        reading its ``interrupted`` property. You may want to save a checkpoint along with the number of the last
        completed batch, so that you can resume from where you left off with ``start_at``.

        If you send a second ``KeyboardInterrupt``, Hydrax will raise a ``KeyboardInterrupt`` at the beginning
        of the next batch. This exception may cause you to lose progress unless you or a framework takes care
        to save a checkpoint in response.

        If you send a third ``KeyboardInterrupt``, the Python interpreter is immediately stopped and control is
        returned to you. You will lose all progress since the last checkpoint.
    """

    __slots__ = (
        "_loader", "_start_at", "_end_at", "_interleave", "_shuffle", "_seed", "_timeout", "_tgroups", "_buffersz",
        "_maxbatch", "_vmode", "_vinterval", "_vgroups", "_vepoch", "_vbatch", "_batches_per_validation", "_interrupt",
        "_abort", "_failed", "_setup", "_running", "_validating", "_watchdog_armed", "_idle_usec", "_first_batch",
        "_tgroup", "_mpctx", "_batches", "_buffers", "_memories", "_submission_id", "_submission_queue",
        "_completion_queue", "_inflight", "_submission_thread", "_completion_thread", "_watchdog_thread", "_loaders",
        "_chain_to", "_batches_per_epoch", "_sigint"
    )

    def __init__(
        self,
        loader_func: Callable[[D, MappingProxyType[str, np.ndarray], int | None], Dict[str, Any] | None],
        training: Iterable[DataGroup[D]] | DataGroup[D],
        *,
        validation: Tuple[str, int, Iterable[DataGroup[D]] | DataGroup[D]] | None = None, # ("epoch" | "batch"), interval, data
        loader_depth: int = 3,
        loader_count: int = 1,
        start_at: Tuple[int, int] = (0, 0), # epoch, epoch batch
        end_at: Tuple[str, int] | None = None, # ("never" | "epoch" | "batch"), interval
        interleave_groups: bool = True,
        shuffle_groups: str = "later", # ("none" | "later" | "all")
        seed: int = 0,
        timeout_sec: int = 60
    ):
        if type(self) is not Dataloader:
            warn(
                f"{type(self).__name__} derives from hydrax.Dataloader. This is not supported.",
                SyntaxWarning
            )

        if is_worker():
            raise Exception("Dataloader cannot run in a worker process.")

        if shuffle_groups not in ("none", "later", "all"):
            raise ValueError("shuffle_groups must be 'none', 'later', or 'all'")

        self._loader = loader_func
        self._end_at = end_at if end_at is not None else ("never", -1)
        self._interleave = interleave_groups
        self._shuffle = shuffle_groups
        self._seed = seed
        self._timeout = timeout_sec if timeout_sec > 0 else 31536000 # okay, not technically disabled, but this is 1 year.

        if self._end_at[0] not in ('never', 'epoch', 'batch'):
            raise ValueError("end_at endpoint must be 'never', 'epoch', or 'batch'")

        self._tgroups = _as_groups(training, 0)
        self._batches_per_epoch = sum(group.data.batches for group in self._tgroups)
        self._buffersz = max(group.data.memory_size for group in self._tgroups)
        self._maxbatch = max(group.data.batch_size for group in self._tgroups)

        if len(self._tgroups) == 1:
            self._interleave = False

        if not self._interleave:
            self._tgroup = -1

        (epochs, batches) = divmod(start_at[1], self._batches_per_epoch)
        self._start_at = (start_at[0] + epochs, batches)

        if self._start_at[0] < 0 or self._start_at[1] < 0:
            raise ValueError("start_at must not be negative")

        if validation is not None:
            if not validation[0] in ("epoch", "batch"):
                raise ValueError("validation mode must be 'epoch' or 'batch'")

            self._vmode = validation[0]
            self._vinterval = validation[1]
            self._vgroups = _as_groups(validation[2], len(self._tgroups))
            self._vepoch = -1
            self._vbatch = -1

            self._buffersz = max(self._buffersz, max(group.data.memory_size for group in self._vgroups))
            self._batches_per_validation = sum(group.data.batches for group in self._vgroups)
        else:
            self._vmode = "none"
            self._batches_per_validation = 0

        self._interrupt = False
        self._abort = False
        self._failed = False

        self._setup = False
        self._running = False
        self._validating = False
        self._watchdog_armed = False

        self._idle_usec = 0
        self._first_batch = True

        self._mpctx = multiprocessing.get_context("spawn")
        self._batches = queue.SimpleQueue[Batch[D] | None]()
        self._buffers = queue.SimpleQueue[SharedMemory]()
        self._memories: List[SharedMemory | None] = [None for _ in range(loader_depth)]

        self._submission_id = 0
        self._submission_queue = self._mpctx.SimpleQueue() # type: multiprocessing.SimpleQueue[Tuple[int, D, str, int, int, int | None] | None]
        self._completion_queue = self._mpctx.SimpleQueue() # type: multiprocessing.SimpleQueue[Tuple[int, Dict[str, Any] | None] | None]
        self._inflight: Dict[int, Tuple[_BatchLoader, int]] = {}

        self._submission_thread: Thread | None = None
        self._completion_thread: Thread | None = None
        self._watchdog_thread: Thread | None = None
        self._loaders: List[Process | None] = [None for _ in range(loader_count)]

        self._chain_to: _BatchLoader | None = None
        self._sigint: Any = None

    def __enter__(self) -> Self:
        assert(not self._setup)
        assert(not self._failed)

        self._setup = True
        self._running = True
        self._validating = False
        self._interrupt = False
        self._watchdog_armed = True

        self._sigint = signal(SIGINT, self._handle_sigint)
        self._idle_usec = 0
        self._first_batch = True

        if not self._interleave:
            self._tgroup = 0

        if self._vmode != "none":
            self._vepoch = 0
            self._vbatch = 0

        for i in range(len(self._memories)):
            memory = SharedMemory(create=True, size=self._buffersz)
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

        self._watchdog_thread = Thread(
            target=self._run_watchdog,
            name="hydrax-watchdog",
            daemon=True
        )
        self._watchdog_thread.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        assert(self._setup)

        if self._failed:
            self._setup = False
            self._running = False
            self._restore_sigint()
            return

        if exc_type is not None:
            print(f"[hydrax]: shutting down due to {exc_type.__name__}")
            self._timeout = 10

        if self._running:
            self._interrupt = True

            try:
                while self._get_batch() is not None:
                    pass
            except:
                self._failed = True
                self._setup = False
                self._running = False
                self._restore_sigint()
                return

            self._running = False

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
        self._restore_sigint()

    def _restore_sigint(self) -> None:
        signal(SIGINT, self._sigint)
        self._sigint = None

    def __iter__(self) -> Self:
        assert(self._running)
        return self

    def __next__(self) -> Batch[D]:
        assert(self._running)
        assert(not self._failed)

        try:
            batch = self._batches.get_nowait()
        except queue.Empty:
            start = time.monotonic_ns()
            batch = self._get_batch()
            end = time.monotonic_ns()

            if batch is not None and not self._first_batch:
                self._idle_usec += (end - start) // 1000

        self._first_batch = False

        if batch is None:
            self._running = False
            raise StopIteration

        if self._interrupt:
            while self._get_batch() is not None:
                pass

            self._running = False
            raise StopIteration

        return batch

    def _get_batch(self) -> Batch[D] | None:
        for _ in range(self._timeout):
            try:
                batch = self._batches.get(timeout=1)
                has_batch = True
            except queue.Empty:
                has_batch = False

            if self._abort:
                self._failed = True
                raise KeyboardInterrupt

            if has_batch:
                return batch

        self._failed = True
        raise BatchTimeoutError(f"no batches have been produced in {self._timeout} seconds")

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
    def interrupted(self) -> bool:
        """``True`` if this dataloader has been interrupted, and ``False`` otherwise."""

        return self._interrupt

    @property
    def first_batch(self) -> int:
        """The index of the first batch to load.

        Controlled by the ``start_at`` argument.
        """

        return self._start_at[0] * self._batches_per_epoch + self._start_at[1]

    @property
    def last_batch(self) -> int | None:
        """The index of the batch to end at, or ``None`` if no end point was specified.

        Controlled by the ``end_at`` argument.
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

        :mod:`hydrax.tqdm` consumes this metric if ``report_interval`` is specified.
        """

        value = self._idle_usec
        self._idle_usec = 0
        return value

    def interrupt(self):
        """Interrupts the dataloader, so no further batches are returned by :func:`__next__`."""

        assert(self._setup)
        self._interrupt = True

    def _load(
        self,
        loader: _BatchLoader[D],
        batch_idx: int,
        data_idx: int,
        seed: int | None
    ) -> None:
        sid = self._submission_id
        self._submission_id += 1

        self._inflight[sid] = (loader, batch_idx)

        try:
            self._submission_queue.put((
                sid,
                loader.group.data._data[data_idx],
                loader.memory.name,
                loader.group.groupid,
                batch_idx,
                seed
            ))
        except:
            traceback.print_exc()
            loader.load_failed(batch_idx)

    def _run_submission(self) -> None:
        self._submit_batches()

        self._chain_to = None
        self._watchdog_armed = False

        for _ in range(len(self._loaders)):
            self._submission_queue.put(None)

        for idx in range(len(self._loaders)):
            self._loaders[idx].join() # type: ignore[union-attr]
            self._loaders[idx] = None

        self._completion_queue.put(None)
        self._completion_thread.join() # type: ignore[union-attr]
        self._completion_thread = None

        self._watchdog_thread.join() # type: ignore[union-attr]
        self._watchdog_thread = None

        gc.collect()

        for _ in range(len(self._memories)):
            for _ in range(5):
                try:
                    buffer = self._buffers.get(timeout=1)
                    break
                except queue.Empty:
                    buffer = None
                    gc.collect()

            if buffer is None:
                self._failed = True
                print("[hydrax]: timed out while collecting batch memory")
                break

        self._batches.put(None)

    def _submit_batches(self) -> None:
        (epoch, seek) = self._start_at
        batch = epoch * self._batches_per_epoch

        while True:
            self._init_groups(epoch)
            ebatch = 0

            while (group := self._select_group()) is not None:
                while (indices := group.reserve()) is not None:
                    if seek == 0:
                        if not self._chain_load(
                            group,
                            TrainingBatch(
                                group.data,
                                indices,
                                epoch,
                                ebatch,
                                batch,
                                self._seed + (batch * self._maxbatch)
                            )
                        ):
                            return
                    else:
                        seek -= 1

                    ebatch += 1
                    batch += 1

                    if seek == 0 and self._vmode == "batch" and batch % self._vinterval == 0:
                        if not self._submit_validation():
                            return

                    if self._end_at[0] == "batch" and batch == self._end_at[1]:
                        return

            epoch += 1

            if seek == 0 and self._vmode == "epoch" and epoch % self._vinterval == 0:
                if not self._submit_validation():
                    return

            if self._end_at[0] == "epoch" and epoch == self._end_at[1]:
                return

    def _init_groups(self, epoch: int) -> None:
        if self._shuffle == "none" or (self._shuffle == "later" and epoch == 0):
            for group in self._tgroups:
                group.reset()
        else:
            rng = Generator(PCG64(self._seed + epoch))
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
                if not self._chain_load(
                    vgroup,
                    ValidationBatch(
                        vgroup.data,
                        indices,
                        self._vepoch,
                        vebatch,
                        self._vbatch
                    )
                ):
                    return False

                vebatch += 1
                self._vbatch += 1

        self._vepoch += 1
        return True

    def _chain_load(self, group: _GroupState[D], batch: Batch[D]) -> bool:
        shm = None

        while shm is None:
            try:
                shm = self._buffers.get(timeout=1)
            except queue.Empty:
                pass

            if self._interrupt:
                if shm is not None:
                    self._buffers.put(shm)

                return False

        loader = _BatchLoader(self, batch, group, shm)
        batch._load(loader)

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
                result = None

    def _run_watchdog(self) -> None:
        multiprocessing.connection.wait(
            [loader.sentinel for loader in self._loaders if loader is not None]
        )

        if self._watchdog_armed:
            print("[hydrax] loader process exited, aborting")
            self._abort = True

    def _handle_sigint(self, signum, frame) -> None:
        if self._abort:
            print("[hydrax] KeyboardInterrupt: terminating")
            os._exit(1)
        elif self._interrupt:
            print("[hydrax] KeyboardInterrupt: aborting, repeat to terminate")
            self._abort = True
            self._watchdog_armed = False
        else:
            print("[hydrax] KeyboardInterrupt: interrupting, repeat to abort")
            self.interrupt()

def _run_loader(
    loader: Callable[[D, MappingProxyType[str, np.ndarray], int | None], Dict[str, Any] | None],
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
                buffer = _LoaderMemory(memories[memory_name])

                result = loader(
                    data,
                    MappingProxyType({
                        key: np.frombuffer(buffer, dtype, count, offset).reshape(shape)[batch_idx] # type: ignore[call-overload]
                        for (key, (shape, dtype, offset, count)) in group_shapes[group_id].items()
                    }),
                    seed
                )

                try:
                    if result is None:
                        result = {}
                    elif not isinstance(result, dict):
                        raise ValueError(f"loader produced a non-dict result of type: {type(result).__name__}")

                    for key, value in result.items():
                        try:
                            if isinstance(value, np.ndarray) or isinstance(value, jax.Array):
                                warn(
                                    f"loader produced an array type with key: '{key}' ({type(value).__name__})",
                                    RuntimeWarning
                                )
                        finally:
                            del value

                    buffer.freeze()

                    completion_queue.put((sid, result))
                finally:
                    del result, buffer
            except:
                completion_queue.put((sid, None))

                print(f"[hydrax] exception while loading data item: {data}")
                traceback.print_exc()

    finally:
        gc.collect()

        for memory in memories.values():
            memory.close()
