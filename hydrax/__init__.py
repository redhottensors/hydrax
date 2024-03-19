"""The main Hydrax module, providing :class:`Dataloader`, :class:`DataGroup`, and :class:`Batch`."""

import gc
import json
import mmap
import os
import pickle
import queue
import time
import traceback

from signal import signal, SIGINT, SIG_IGN
from threading import Thread, BoundedSemaphore, Lock
from types import MappingProxyType
from warnings import warn

import multiprocessing as mp
from multiprocessing import Process
from multiprocessing.shared_memory import SharedMemory

from typing import (
    Iterable, Iterator, Mapping, Sequence, Callable, Generic, Any, TypeVar, TypeAlias,
    Optional, cast, final, TYPE_CHECKING
)

import numpy as np
from numpy.random import Generator, PCG64

try:
    from numpy.typing import DTypeLike
except ImportError:
    if not TYPE_CHECKING:
        DTypeLike: TypeAlias = Any

import jax # type: ignore[import-not-found]
import jax.numpy as jnp # type: ignore[import-not-found]

from ._trackedbuffer import TrackedBuffer

D = TypeVar('D')
ArraySpec: TypeAlias = tuple[tuple[int, ...], DTypeLike, jax.typing.DTypeLike]
ArrayLayout: TypeAlias = tuple[tuple[int, ...], np.dtype, jnp.dtype, int, int]
StartupFunc: TypeAlias = Callable[[], None]
LoaderFunc: TypeAlias = Callable[[D, MappingProxyType[str, np.ndarray], int | None], dict[str, Any] | None]
PlacementFunc: TypeAlias = Callable[
    [
        MappingProxyType[str, np.ndarray],
        MappingProxyType[str, ArrayLayout],
        MappingProxyType[str, Sequence[Any]],
        "DataGroup[D]",
    ],
    dict[str, jax.Array]
]

_Submission: TypeAlias = tuple[int, D, str, int, int, int | None]
_Completion: TypeAlias = tuple[int, dict[str, object] | None]
_CacheSave: TypeAlias = tuple["Batch[D]", Optional["_CacheBatchMemory"]]
_SubarrayLayout: TypeAlias = tuple[str, tuple[int, ...], np.dtype, int, int, int]

_O_DEFAULT = (
    getattr(os, "O_BINARY", 0)    |
    getattr(os, "O_CLOEXEC", 0)   |
    getattr(os, "O_NOINHERIT", 0) |
    getattr(os, "O_NOATIME", 0)   |
    getattr(os, "O_NOCTTY", 0)
)

if hasattr(mmap, "MAP_POPULATE"):
    _MMAP_OPEN = { "flags": mmap.MAP_SHARED | mmap.MAP_POPULATE, "prot": mmap.PROT_READ }
else:
    _MMAP_OPEN = { "access": mmap.ACCESS_READ }

_MADV_NOSYNC = getattr(mmap, "MADV_NOSYNC", 0)

T = TypeVar('T')
def _nonnull(value: T | None) -> T:
    assert value is not None
    return value

@final
class BatchTimeoutError(TimeoutError):
    """Raised if no batch has been produced for some time.

    Controlled by the ``timeout_sec`` argument to :class:`Dataloader`.
    """

def is_worker() -> bool:
    """Returns ``True`` if running in a subprocess.

    Subprocesses cannot instantiate :class:`Dataloader`.
    """

    return mp.parent_process() is not None

def _as_seq(obj: Iterable[D]) -> Sequence[D]:
    if isinstance(obj, Sequence):
        return obj

    if isinstance(obj, Iterable):
        return list(obj)

    raise TypeError(f"{type(obj).__name__} is not a sequence or iterable")

class _BatchMemory(TrackedBuffer):
    __slots__ = ("rc",)

    def __init__(self, _mem: Any):
        self.rc = 0

    def __repr__(self) -> str:
        return f"<{type(self).__name__} at 0x{id(self):x} rc={self.rc}, recycled={self.recycled}, buffer={repr(self.buffer)}>"

    def _ref(self) -> None:
        if self.recycled:
            raise RuntimeError("batch memory has been recycled")

        self.rc += 1

    def _deref(self) -> None:
        self.rc -= 1
        if self.rc <= 0:
            self.recycle()

    @property
    def recycled(self) -> bool:
        raise NotImplementedError

    @property
    def name(self) -> str:
        raise NotImplementedError

    def release(self) -> None:
        if self.rc == 0 and not self.recycled:
            self.recycle()

    def recycle(self) -> None:
        raise NotImplementedError

    def make_arrays(self, layouts: dict[str, ArrayLayout]) -> dict[str, np.ndarray]:
        return {
            name: np.frombuffer(cast(Any, self), np_dtype, count, offset).reshape(shape)
            for name, (shape, np_dtype, jax_dtype, offset, count) in layouts.items()
        }

class _SharedBatchMemory(_BatchMemory):
    __slots__ = ("_dataloader", "_shm")

    def __new__(cls, dataloader: "Dataloader", shm: SharedMemory):
        return super().__new__(cls, shm.buf) # type: ignore [arg-type]

    def __init__(self, dataloader: "Dataloader", shm: SharedMemory):
        super().__init__(shm.buf)

        self._dataloader = dataloader
        self._shm: SharedMemory | None = shm

    @property
    def recycled(self) -> bool:
        return self._shm is None

    @property
    def name(self) -> str:
        if self._shm is None:
            raise RuntimeError("batch memory has been recycled")

        return self._shm.name

    def recycle(self) -> None:
        assert self.rc == 0
        assert self._shm is not None

        self._dataloader._buffers.append(self._shm)
        self._dataloader._semaphore.release()

        self._shm = None

class _CacheBatchMemory(_BatchMemory):
    __slots__ = ("_dataloader", "_fd", "_name", "_abandoned")

    def __new__(cls, dataloader: "Dataloader", fd: int | None, mem: mmap.mmap, name: str):
        return super().__new__(cls, mem) # type: ignore[arg-type]

    def __init__(self, dataloader: "Dataloader", fd: int | None, mem: mmap.mmap, name: str):
        super().__init__(mem)

        self._dataloader = dataloader
        self._fd = fd
        self._name: str | None = name
        self._abandoned = False

    @property
    def recycled(self) -> bool:
        return self._name is None

    @property
    def name(self) -> str:
        if self._name is None:
            raise RuntimeError("batch memory has been recycled")

        return self._name

    def recycle(self) -> None:
        assert self.rc == 0
        assert self._name is not None

        self._name = None
        self.buffer.close()

        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

        if not self._abandoned:
            self._dataloader._semaphore.release()

    def abandon(self) -> None:
        self._abandoned = True

        if self.rc == 0:
            self.recycle()

    def take_fd(self) -> int:
        fd = _nonnull(self._fd)
        self._fd = None

        return fd

    def make_arrays(self, layouts: dict[str, ArrayLayout]) -> dict[str, np.ndarray]:
        arrays = super().make_arrays(layouts)

        return arrays

    @staticmethod
    def create(dataloader: "Dataloader[D]", batch: "Batch[D]") -> "_CacheBatchMemory":
        path = _nonnull(batch._cache_path)
        size = _nonnull(batch._group.data._cache_size)

        fd = os.open(path, os.O_RDWR | os.O_CREAT | _O_DEFAULT, 0o666)
        try:
            os.ftruncate(fd, size)

            mem = mmap.mmap(fd, size, access = mmap.ACCESS_READ)
            try:
                return _CacheBatchMemory(dataloader, fd, mem, path)
            except:
                mem.close()
                raise
        except:
            os.close(fd)
            raise

    @staticmethod
    def open(dataloader: "Dataloader[D]", batch: "Batch[D]") -> "_CacheBatchMemory":
        path = _nonnull(batch._cache_path)

        fd = os.open(path, os.O_RDONLY | _O_DEFAULT)
        try:
            mem = mmap.mmap(fd, 0, **_MMAP_OPEN)
            try:
                return _CacheBatchMemory(dataloader, None, mem, path)
            except:
                mem.close()
                raise
        finally:
            os.close(fd)

class _LoaderMemory(TrackedBuffer):
    __slots__ = ("_rc", "_frozen")

    def __init__(self, _buffer: Any):
        self._rc = 0
        self._frozen = False

    def __repr__(self) -> str:
        return f"<{type(self).__name__} at 0x{id(self):x} rc={self._rc}, frozen={self._frozen}, buffer={repr(self.buffer)}>"

    def _ref(self) -> None:
        if self._frozen:
            raise RuntimeError("loader memory has been transferred")

        self._rc += 1

    def _deref(self) -> None:
        self._rc -= 1

        if self._rc == 0 and self._frozen:
            self.release()

    def freeze(self) -> None:
        self._frozen = True

        if self._rc > 0:
            warn("loader memory has dangling reference", RuntimeWarning)
        else:
            self.release()

    def release(self) -> None:
        raise NotImplementedError

class _SharedLoaderMemory(_LoaderMemory):
    __slots__ = ("_layouts", "_index")

    def __new__(cls, shm: SharedMemory, _layouts: dict[str, ArrayLayout], _index: int):
        return super().__new__(cls, shm.buf) # type: ignore[arg-type]

    def __init__(self, shm: SharedMemory, layouts: dict[str, ArrayLayout], index: int):
        super().__init__(shm.buf)
        self._layouts = layouts
        self._index = index

    def release(self) -> None:
        pass

    def make_arrays(self) -> dict[str, np.ndarray]:
        return {
            name: np.frombuffer(cast(Any, self), np_dtype, count, offset).reshape(shape)[self._index]
            for name, (shape, np_dtype, jax_dtype, offset, count) in self._layouts.items()
        }

class _CacheLoaderMemory(_LoaderMemory):
    __slots__ = ("name", "_shape", "_offset", "_dtype", "_count")

    def __new__(cls, fd: int, layout: _SubarrayLayout, index: int):
        _, _, _, _, base, size = layout

        offset = base + size * index
        align = offset % mmap.ALLOCATIONGRANULARITY

        mem = mmap.mmap(fd, align + size, offset=offset - align)
        try:
            if _MADV_NOSYNC != 0:
                mem.madvise(_MADV_NOSYNC)

            self = super().__new__(cls, mem) # type: ignore[arg-type]
            self._offset = align
            return self
        except:
            mem.close()
            raise

    def __init__(self, _fd: int, layout: _SubarrayLayout, _index: int):
        super().__init__(self.buffer)
        self.name, self._shape, self._dtype, self._count, _, _ = layout

        if TYPE_CHECKING:
            self._offset = -1

    def release(self) -> None:
        self.buffer.close()

    def make_array(self) -> np.ndarray:
        return np.frombuffer(cast(Any, self), self._dtype, self._count, self._offset).reshape(self._shape)

class _CacheLoaderMemories:
    __slots__ = ("_memories",)

    def __init__(self, path: str, layouts: list[_SubarrayLayout], index: int):
        fd = os.open(path, os.O_RDWR | _O_DEFAULT)
        self._memories: list[_CacheLoaderMemory] = []

        try:
            for layout in layouts:
                self._memories.append(_CacheLoaderMemory(fd, layout, index))
        except Exception:
            for memory in self._memories:
                memory.release()

            self._memories.clear()
            raise
        finally:
            os.close(fd)

    def freeze(self) -> None:
        for memory in self._memories:
            memory.freeze()

        self._memories.clear()

    def make_arrays(self) -> dict[str, np.ndarray]:
        return { memory.name: memory.make_array() for memory in self._memories }

@final
class DataGroup(Sequence[D], Generic[D]):
    """Represents a group of data which share the same descriptor, batch size, and array shapes.

    .. caution::
        Don't derive from DataGroup, and do not modify your dataset after placing it in a DataGroup.

    :param batch_size: The batch size for loading and processing the data.
    :param data: A list of all data descriptors for this group. Descriptors are passed to the loader to identify the
        data item to load. Any finite sequence-like object or iterator is acceptable. The elements must pickleable, as
        they are sent directly to loader processes.
    :param loader_arrays: Shape and datatype definitions for all arrays in a loader batch. Do not include the leading
        batch dimension, since that is specified by ``batch_size``. These arrays will be presented to the loader for
        zero-copy initialization.
    :param cache_arrays: Shape and datatype definitions for all arrays in a cached batch. Do not include the leading
        batch dimension, since that is specified by ``batch_size``. Arrays of this shape will be retrieved from and
        stored to disk. If not specified, ``loader_arrays``  and data is automatically cached. Otherwise, the data to
        cache must be provided via :func:`Batch.cache`.
    :param cache_location: The location of the cache on-disk. If the path does not exist, it is created unless
        ``cache_readonly`` is specified.
    :param cache_readonly: If ``True``, the cache is readonly and will not be created if it does not exist, nor will it
        be populated if batches are missing. The default is ``False`` to allow creation and population.
    :param seed: Integer seed to use for shuffling and batch seeding. The default is ``0``.
    :param shuffle_first: If ``True``, the first epoch is shuffled itemwise. Otherwise, the first epoch proceeds in the
        order specified in the dataset, which is the default.
    :param shuffle_later: A string indicating the shuffling and seeding mode for epochs after the first. The default is
        ``"default"``, see below.

        - ``"repeat"`` - do not shuffle or reseed batches
        - ``"reseed"`` - reseed batches but do not shuffle
        - ``"itemwise"`` - reseed batches and shuffle items between them
        - ``"batchwise"`` - shuffle the order of batches, but do not reseed or change their contents
        - ``"default"`` - ``"batchwise"`` if ``cache_location`` is specified, otherwise ``"itemwise"``

    If you do not specify ``loader_arrays``, any batches which cannot be loaded from the cache will be dropped. As such,
    both ``cached_arrays`` and ``cache_location`` are required, and ``cache_readonly`` is implied.

    .. warning::
        If your dataset is an iterable and not otherwise indexable, it will be materialized by the DataGroup. If you
        have hundreds of thousands of items, consider using the :mod:`hydrax.pandas` adapter module.

        A cache cannot be reused if the ``batch_size``, ``data``, ``cache_arrays``, ``seed``, or shuffling parameters
        have changed.  Only the length of the ``data`` is verified, not its ordering or contents. If either of those
        change, the result of using the cache is undefined. You may change the name and shape of the cache arrays as
        long as their position, NumPy dtype, and total size remain the same.
    """

    __slots__ = (
        "_data", "_batch_size", "_batch_count", "_loader_layouts", "_loader_size", "_cache_layouts", "_cache_size",
        "_cache_location", "_cache_readonly", "_cache_direct", "_seed", "_shuffle_first", "_shuffle_later"
    )

    def __init__(
        self,
        batch_size: int,
        data: Iterable[D],
        *,
        loader_arrays: Mapping[str, ArraySpec] | None = None,
        cache_arrays: Mapping[str, ArraySpec] | None = None,
        cache_location: str | os.PathLike | None = None,
        cache_readonly: bool = False,
        seed: int = 0,
        shuffle_first: bool = False,
        shuffle_later: str = "default"
    ):
        if type(self) is not DataGroup: # pylint: disable=unidiomatic-typecheck
            warn(f"{type(self).__name__} derives from hydrax.DataGroup. This is not supported.", SyntaxWarning)

        if shuffle_later not in ("default", "repeat", "reseed", "itemwise", "batchwise"):
            raise ValueError("shuffle_later must be 'default', 'repeat', 'reseed', 'itemwise', or 'batchwise'")

        self._data = _as_seq(data)
        self._batch_size = min(len(self._data), batch_size)
        self._batch_count = len(self._data) // self._batch_size
        self._cache_direct = False

        if self._batch_size < 1:
            raise ValueError("batch size must be at least 1")

        if loader_arrays is None:
            if cache_arrays is None:
                raise ValueError("either loader_arrays or cache_arrays must be specified")

            if cache_location is None:
                raise ValueError("cache_location is required if no loader_arrays are specified")

            cache_readonly = True

        if cache_location is None:
            cache_readonly = True

            if shuffle_later == "default":
                shuffle_later = "itemwise"
        else:
            if cache_arrays is None:
                cache_arrays = loader_arrays

                if not cache_readonly:
                    self._cache_direct = True

            if shuffle_later == "default":
                shuffle_later = "batchwise"

        self._loader_layouts, self._loader_size = self._layout(self._batch_size, loader_arrays)
        self._cache_layouts, self._cache_size = self._layout(self._batch_size, cache_arrays)
        self._cache_location = cache_location
        self._cache_readonly = cache_readonly

        self._seed = seed
        self._shuffle_first = shuffle_first
        self._shuffle_later = shuffle_later

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx):
        return self._data[idx]

    @property
    def batch_size(self) -> int:
        """The batch size configured for the DataGroup."""

        return self._batch_size

    @property
    def batch_count(self) -> int:
        """The total number of batches this DataGroup can produce per epoch."""

        return self._batch_count

    @property
    def loader_layouts(self) -> MappingProxyType[str, ArrayLayout] | None:
        """The array layouts for all arrays provided to the loader. Includes the leading batch dimension.

        The mapping has the form:
        ``{ 'array_name': ((batch_size, dim_1, ...), numpy_dtype, jax_dtype, offset, count), ... }``
        """

        if self._loader_layouts is None:
            return None

        return MappingProxyType(self._loader_layouts)

    @property
    def cache_layouts(self) -> MappingProxyType[str, ArrayLayout] | None:
        """The array layouts for all arrays in the cache. Includes the leading batch dimension.

        The mapping has the form:
        ``{ 'array_name': ((batch_size, dim_1, ...), numpy_dtype, jax_dtype, offset, count), ... }``
        """

        if self._cache_layouts is None:
            return None

        return MappingProxyType(self._cache_layouts)

    @property
    def loader_size(self) -> int | None:
        """The size, in bytes, of a single loaded batch, or ``None`` if ``loader_arrays`` was not specified."""
        return self._loader_size

    @property
    def cache_size(self) -> int | None:
        """The size, in bytes, of a single cached batch, or ``None`` if ``cached_arrays`` was not specified."""
        return self._cache_size

    def _prepare(self) -> None:
        if self._cache_location is None:
            return

        path = os.path.join(self._cache_location, "cache.json")
        expected = json.dumps({
            "_type": "hydrax-cache-v1",
            "data_len": len(self._data),
            "batch_size": self._batch_size,
            "seed": self._seed,
            "shuffle_first": self._shuffle_first,
            "shuffle_later": self._shuffle_later,
            "array_layouts": [
                { "offset": offset, "dtype": str(np_dtype), "count": count }
                for (_shape, np_dtype, _jax_dtype, offset, count) in sorted(
                    _nonnull(self._cache_layouts).values(),
                    key=lambda layout: layout[3]
                )
            ]
        }, indent=4)

        if self._cache_readonly:
            mode = "r"
        else:
            mode = "a+"
            os.makedirs(os.path.abspath(self._cache_location), exist_ok=True)

        with open(path, mode, encoding="utf-8", newline="") as file:
            if not self._cache_readonly:
                file.seek(0)

            found = file.read()

            if not self._cache_readonly and found == "":
                file.write(expected)
                file.flush()
                os.fsync(file.fileno())
                return

        if found != expected:
            raise RuntimeError(f"cache descriptor \"{path}\" does not match configuration")

    @staticmethod
    def _layout(
        batch_size: int,
        array_defs: Mapping[str, ArraySpec] | None
    ) -> tuple[dict[str, ArrayLayout] | None, int | None]:
        if array_defs is None:
            return None, None

        layouts = { }
        size = 0

        for (key, (shape, np_dtype, jax_dtype)) in array_defs.items():
            np_dtype = np.dtype(np_dtype)
            jax_dtype = jnp.dtype(jax_dtype)
            shape = (batch_size, *shape)
            count = np.prod(shape).item()

            if count < 1:
                raise ValueError(f"invalid shape for array '{key}': {shape}")

            size += (-size) % np_dtype.alignment
            layouts[key] = (shape, np_dtype, jax_dtype, size, count)
            size += count * np_dtype.itemsize

        return layouts, size

def _check_jax_arrays(arrays: dict[str, jax.Array], layouts: dict[str, ArrayLayout]) -> None:
    if not __debug__:
        return

    if not isinstance(arrays, dict):
        raise TypeError(f"arrays has non-dict type {type(arrays).__name__}")

    if len(arrays) < len(layouts):
        for name in layouts:
            if not name in arrays:
                raise ValueError(f"array '{name}' was not provided")

    for name, array in arrays.items():
        if not isinstance(array, jax.Array):
            raise TypeError(f"array '{name}' of type {type(array).__name__} is not a JAX array")

        expected = layouts.get(name)
        if expected is None:
            raise ValueError(f"array '{name}' has unexpected name")

        if expected[0] != array.shape:
            raise ValueError(f"array '{name}' has shape {array.shape}, not {expected[0]}")

        if expected[2] != array.dtype:
            raise ValueError(f"array '{name}' has dtype {array.dtype}, not {expected[2]}")

def _check_additional_data(additional: dict[str, list[object]], batch_size: int) -> None:
    if not __debug__:
        return

    if not isinstance(additional, dict):
        raise TypeError(f"additional data has non-dict type {type(additional).__name__}")

    for name, data in additional.items():
        if not isinstance(data, list):
            raise TypeError(f"additional data '{name}' of type {type(data).__name__} is not a list")

        if len(data) != batch_size:
            raise ValueError(f"additional data '{name}' has length {len(data)}, expected {batch_size}")

        for idx, value in enumerate(data):
            if isinstance(value, (np.ndarray, jax.Array)):
                warn(f"additional data '{name}' value {idx} has array type {type(value).__name__}", RuntimeWarning)

def _check_item_additional_data(additional: dict[str, object]) -> None:
    if not __debug__:
        return

    if not isinstance(additional, dict):
        raise TypeError(f"additional data has non-dict type {type(additional).__name__}")

    for key, value in additional.items():
        if isinstance(value, (np.ndarray, jax.Array)):
            warn(f"additional data '{key}' has array type {type(value).__name__}", RuntimeWarning)

class _Items(Sequence):
    __slots__ = ("_array",)

    def __init__(self, array: np.ndarray):
        self._array = array

    def __len__(self) -> int:
        return self._array.size

    def __getitem__(self, key):
        return self._array.item(key)

class _GroupState(Generic[D]):
    __slots__ = ("data", "groupid", "dataloader", "_item_indices", "_batch_indices", "_cursor", "_used")

    def __init__(self, dataloader: "Dataloader[D]", data: DataGroup[D], groupid: int):
        if not isinstance(data, DataGroup):
            raise TypeError(f"data of type {type(data).__name__} is not in a hydrax.DataGroup")

        self.data = data
        self.groupid = groupid
        self.dataloader = dataloader

        self._batch_indices: np.ndarray | None = None
        self._item_indices: np.ndarray | None = None
        self._cursor = 0
        self._used = 0.0

    @property
    def used(self) -> float:
        return self._used

    def prepare(self, epoch: int) -> None:
        if self.data._shuffle_first and not (self.data._shuffle_later == "itemwise" and epoch > 0):
            rng = Generator(PCG64(self.data._seed))
            self._item_indices = rng.permutation(len(self.data))

    def shuffle(self, epoch: int) -> None:
        self._cursor = 0
        self._used = 0.0

        if epoch == 0:
            return

        match self.data._shuffle_later:
            case "repeat" | "reseed":
                pass

            case "itemwise":
                rng = Generator(PCG64(self.data._seed + epoch))
                self._item_indices = rng.permutation(len(self.data))

            case "batchwise":
                rng = Generator(PCG64(self.data._seed + epoch))
                self._batch_indices = rng.permutation(self.data.batch_count)

            case _:
                raise AssertionError

    def reserve_training(self, epoch: int) -> tuple[int, Sequence[int], int] | None:
        if (reservation := self.reserve_validation()) is None:
            return None

        batch_index, indices = reservation

        match self.data._shuffle_later:
            case "reseed" | "itemwise":
                batch_seed = (
                    self.data._seed +
                    (epoch * self.data.batch_count * self.data.batch_size) +
                    (batch_index * self.data.batch_size)
                )

            case "repeat" | "batchwise":
                batch_seed = (
                    self.data._seed +
                    (batch_index * self.data.batch_size)
                )

            case _:
                raise AssertionError

        return batch_index, indices, batch_seed

    def reserve_validation(self) -> tuple[int, Sequence[int]] | None:
        if self._cursor >= self.data.batch_count:
            return None

        if self._batch_indices is None:
            batch_index = self._cursor
        else:
            batch_index = self._batch_indices.item(self._cursor)

        self._cursor += 1
        self._used = self._cursor / self.data.batch_count

        start = batch_index * self.data.batch_size
        end = start + self.data.batch_size

        if self._item_indices is None:
            return batch_index, range(start, end)
        else:
            return batch_index, _Items(self._item_indices[start:end])

def _as_groups(
    dataloader: "Dataloader[D]",
    groups: Iterable[DataGroup[D]] | DataGroup[D],
    first_id: int
) -> list[_GroupState]:
    if isinstance(groups, DataGroup):
        return [_GroupState(dataloader, groups, first_id)]

    return [_GroupState(dataloader, group, first_id + idx) for (idx, group) in enumerate(groups)]

class _BatchData(Sequence[D]):
    __slots__ = ("_group", "_indices")

    def __init__(self, batch: "Batch[D]"):
        self._group = batch._group
        self._indices = batch._indices

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, key):
        return self._group[self._indices[key]]

class Batch(Generic[D]):
    """An abstract base class representing a batch of data loaded by a :class:`Dataloader`.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.

    Batches are returned by :func:`Dataloader.__next__`.

    If ``is_training`` is ``True``, this is a :class:`TrainingBatch`. Otherwise, ``is_validation`` is ``True`` and this
    instance is a :class:`ValidationBatch`. If you did not opt in to validation on the :class:`Dataloader``, all your
    batches will be ``TrainingBatch``.

    If you are not using :class:`DataGroup` caching, you can ignore the following explanation. All of your Batches will
    be loaded :attr:`uncached` and you cannot call :func:`cache`.

    If you are using :class:`DataGroup` caching and specified both ``loader_arrays`` and ``cache_arrays`` then you will
    receive batches in both the cached and :attr:`uncached` layouts. Even if you do not intend to persist a given Batch,
    you should convert it into the cached form by calling :func:`cache` after transforming your data from its uncached
    layout, for example by applying a fixed encoder network. In future epochs, your cached data will be loaded directly,
    unless you opted out with ``cache_readonly = True`` or ``persist = False``.

    .. attention::
        Do not retain references to a Batch or its :attr:`arrays` beyond one model batch loop, and be careful not to do
        so accidentally. In limited cases, such as asynchronous logging, it is okay to retain references while that is
        occurring, but it's important to ensure all references are released once they are no longer needed. As long as a
        Batch's arrays are referenced (including by the Batch object itself), it consumes one :class:`Dataloader`
        ``depth``. If all of the ``depth`` is consumed in this way the Dataloader will stall (or deadlock, if the
        references are never released).
    """

    __slots__ = ("_group", "_index", "_indices", "_arrays", "_additional", "_uncached", "_cache_path")

    def __init__(self, group: _GroupState[D], epoch: int, index: int, indices: Sequence[int]):
        self._group = group
        self._index = index
        self._indices = indices
        self._arrays: dict[str, jax.Array] = {}
        self._additional: dict[str, list[object]] = {}
        self._uncached = True

        if group.data._cache_location is None:
            self._cache_path = None
        elif group.data._shuffle_later in ("reseed", "itemwise"):
            self._cache_path = os.path.join(group.data._cache_location, f"e{epoch}-b{index}.dat")
        else:
            self._cache_path = os.path.join(group.data._cache_location, f"e0-b{index}.dat")

    def __len__(self) -> int:
        return len(self._indices)

    @property
    def is_validation(self) -> bool:
        """``True`` if this instance is a :class:`ValidationBatch`, and ``False`` otherwise."""

        raise NotImplementedError

    @property
    def is_training(self) -> bool:
        """``True`` if this instance is a :class:`TrainingBatch`, and ``False`` otherwise."""

        raise NotImplementedError

    @property
    def group(self) -> DataGroup[D]:
        """The :class:`DataGroup` from which this batch was loaded."""

        return self._group.data

    @property
    def group_index(self) -> int:
        """The index of this batch within :attr:`group`."""

        return self._index

    @property
    def indices(self) -> Sequence[int]:
        """The corresponding indices in :attr:`group` of the data in this batch."""

        return self._indices

    @property
    def data(self) -> Sequence[D]:
        """The data descriptors of the data in this batch.

        The length of this sequence is equal to the batch size of the :attr:`group`.
        """

        return _BatchData(self)

    def get_data(self, index: int) -> D:
        """Returns the data descriptor for the specified item in this batch.

        :param index: The name of the array, as defined in the `DataGroup`.

        .. seealso::
            This is equivalent to ``data[index]``. See :attr:`data`.
        """

        return self._group.data[self._indices[index]]

    @property
    def layouts(self) -> MappingProxyType[str, ArrayLayout]:
        """The layouts of ``arrays``.

        If this batch is :attr:`uncached`, this is ``group.loader_layouts``, otherwise it is ``group.cache_layouts``.
        """

        if self._uncached:
            return _nonnull(self._group.data.loader_layouts)
        else:
            return _nonnull(self._group.data.cache_layouts)

    @property
    def arrays(self) -> MappingProxyType[str, jax.Array]:
        """The JAX arrays for this batch, as defined by the :class:`DataGroup`.

        The leading dimension of each array is the batch size of the ``DataGroup``. The shapes and dtypes are as
        specified by ``group.loader_layouts`` if the batch is ``uncached``, otherwise they are as specified by
        ``group.cache_layouts``. The current layout is available as the :attr:`layouts` property.

        .. tip::
            To convert an uncached batch to a cached batch, use :func:`cache`.
        """

        return MappingProxyType(self._arrays)

    def get_array(self, name: str) -> jax.Array:
        """Returns the specified JAX array for this batch, as defined in the :class:`DataGroup`.

        :param name: The name of the array, as defined in the `DataGroup`.

        .. seealso::
            This is equivalent to ``arrays[name]``. See :attr:`arrays`.
        """

        return self._arrays[name]

    @property
    def additional(self) -> MappingProxyType[str, Sequence[Any]]:
        """The additional data returned by the loader for each item of the batch.

        This data is a readonly mapping of the form ``{ 'key_0': [value_0, ...], ... }``, where the number of values for
        each key is equal to the batch size of the ``DataGroup``. A value will be ``None`` if the loader did not return
        additional data with the corresponding key for the corresponding item in this batch.

        .. caution:
            If your loader function does not return a given key for all data items that it loads, it is possible you may
            encounter a batch where no data item produces the specified key, and so it is not present in the returned
            mapping. If this is the case, consider using :func:`get_additional` instead.
        """

        return MappingProxyType(self._additional)

    def get_additional(self, key: str, index: int | None = None, *, default: Any = None) -> Any:
        """Returns the additional data returned by the loader with the specified name.

        :param key: The key returned by the loader.
        :param index: The index of the data item within this batch. If this is not specified, a sequence corresponding
            to each item in the batch is returned.
        :param default: The default value to return if the additional data is not present.
        """

        additional = self._additional.get(key)
        if additional is None:
            if index is None:
                return [default for _ in range(len(self._indices))]

            if index < -len(self._indices) or index >= len(self._indices):
                raise IndexError(f"index {index} is invalid for batch size {len(self._indices)}")

            return default

        if index is None:
            return [value if value is not None else default for value in additional]

        value = additional[index]
        return value if value is not None else default

    @property
    def uncached(self) -> bool:
        """``True`` if the batch is in the loader format specified by the :class:`DataGroup`, and ``False`` if it is in
        the cached format.

        .. tip::
            To convert an uncached batch to a cached batch, use :func:`cache`.
        """

        return self._uncached

    def cache(
        self,
        arrays: dict[str, jax.Array],
        additional: dict[str, list[object]] | None = None,
        *,
        persist: bool = True
    ) -> None:
        """Converts an uncached batch into a cached batch by providing the arrays and additional data to cache.

        .. important::
            This method will raise a ``RuntimeError`` if the batch is already cached, or if cache layouts were not
            specified in the :class:`DataGroup`.

        :param arrays: The JAX arrays to cache. These must exactly correspond to :attr:`DataGroup.cache_layouts`.
        :param additional: Optional additional data to cache. The contents must be pickleable and the length of each
            list must be exactly equal to :attr:`DataGroup.batch_size`. If not specified, the batch's current
            :attr:`additional` data is retained.
        :param persist: If ``False`` the batch will not actually be persisted to disk and so will need to be reloaded.
            Additionally, persistence is skipped if ``cache_readonly`` was specified when creating the
            :class:`DataGroup`.
        """

        if not self._uncached:
            raise RuntimeError("batch is already cached")

        layouts = self._group.data._cache_layouts
        if layouts is None:
            raise RuntimeError("cache layouts undefined")

        _check_jax_arrays(arrays, layouts)

        if additional is not None:
            _check_additional_data(additional, self._group.data._batch_size)
            self._additional = additional.copy()

        self._arrays = arrays.copy()
        self._uncached = False

        if persist and not self._group.data._cache_readonly:
            if self._group.dataloader._begin_caching(self):
                self._group.dataloader._cache_save(self, None)

    def _submit_to_loaders(self, inflight: "_InflightLoad[D]") -> None:
        raise NotImplementedError

    def _from_cached(self, arrays: dict[str, jax.Array], additional: dict[str, list[object]]) -> None:
        self._arrays = arrays
        self._additional = additional
        self._uncached = False

    def _item_loaded(self, idx: int, additional: dict[str, Any]) -> None:
        for (key, value) in additional.items():
            if not key in self._additional:
                self._additional[key] = [None for _ in range(len(self))]

            self._additional[key][idx] = value

    def _arrays_loaded(self, arrays: dict[str, jax.Array]) -> None:
        self._arrays = arrays

        if self._group.data._cache_direct:
            self._uncached = False

@final
class TrainingBatch(Batch[D], Generic[D]):
    """A batch of training data loaded by a :class:`Dataloader`.

    Batches are returned by :func:`Dataloader.__next__`. You can determine if a :class:`Batch` is a ``TrainingBatch``
    by checking :attr:`is_training`.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.
    """

    __slots__ = ("_epoch", "_epoch_batch", "_batch_num", "_seed")

    def __init__(
        self,
        group: _GroupState[D],
        index: int,
        indices: Sequence[int],
        epoch: int,
        epoch_batch: int,
        batch_num: int,
        seed: int,
    ):
        super().__init__(group, epoch, index, indices)

        self._epoch = epoch
        self._epoch_batch = epoch_batch
        self._batch_num = batch_num
        self._seed = seed

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

        return self._batch_num

    @property
    def seed(self) -> int:
        """The deterministic seed for randomness associated with this batch."""

        return self._seed

    @property
    def seeds(self) -> Sequence[int]:
        """The deterministic seeds for randomness associated with each item of this batch.

        Each seed is the same seed that was passed to the :class:`Dataloader` ``loader_func`` for the corresponding item
        of this batch.
        """

        return range(self._seed, self._seed + len(self._indices))

    def _submit_to_loaders(self, inflight: "_InflightLoad[D]") -> None:
        for (batch_idx, data_idx) in enumerate(self._indices):
            inflight.chain.dataloader._submit_to_loaders(inflight, batch_idx, data_idx, self._seed + batch_idx)

@final
class ValidationBatch(Batch[D], Generic[D]):
    """A batch of validation data loaded by a :class:`Dataloader`.

    Batches are returned by :func:`Dataloader.__next__`. You can determine if a :class:`Batch` is a ``ValidationBatch``
    by checking :attr:`is_validation`.

    .. caution::
        Don't derive from or instantiate this type, it is created internally.
    """

    __slots__ = ("_vepoch", "_vepoch_batch", "_vbatch")

    def __init__(
        self,
        group: _GroupState[D],
        index: int,
        indices: Sequence[int],
        vepoch: int,
        vepoch_batch: int,
        vbatch: int,
    ):
        super().__init__(group, 0, index, indices)
        self._vepoch = vepoch
        self._vepoch_batch = vepoch_batch
        self._vbatch = vbatch

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

        return self._vepoch

    @property
    def validation_epoch_batch(self) -> int:
        """The zero-based index of this batch within the current validation epoch."""

        return self._vepoch_batch

    @property
    def validation_batch_num(self) -> int:
        """The overall zero-based index of this validation batch.

        Unlike a training batch number, this number starts at zero regardless of how many validation batches were
        skipped by the ``start_at`` argument to :class:`Dataloader`. This number counts separately from training batch
        numbers.
        """

        return self._vbatch

    def _submit_to_loaders(self, inflight: "_InflightLoad[D]") -> None:
        for (batch_idx, data_idx) in enumerate(self._indices):
            inflight.chain.dataloader._submit_to_loaders(inflight, batch_idx, data_idx, None)

class _LoaderChain(Generic[D]):
    __slots__ = ("dataloader", "group", "batch", "_prior_ready", "_batch_ready", "_chained")

    def __init__(self, dataloader: "Dataloader[D]", batch: Batch[D]):
        self.dataloader = dataloader
        self.group = batch._group
        self.batch: Batch[D] | None = batch

        self._prior_ready = False
        self._batch_ready = False
        self._chained: _LoaderChain[D] | None = None

    def prior_ready(self) -> None:
        self._prior_ready = True
        self._check_chain()

    def batch_ready(self) -> None:
        self._batch_ready = True
        self._check_chain()

    def batch_failed(self) -> None:
        self.batch = None
        self._batch_ready = True
        self._check_chain()

    def chain_to(self, chained: "_LoaderChain[D]") -> None:
        self._chained = chained
        self._check_chain()

    def _check_chain(self) -> None:
        if not self._prior_ready or not self._batch_ready:
            return

        if self.batch is None and self._chained is None:
            return

        with self.dataloader._chain_lock:
            if self.batch is not None:
                self.dataloader._batches.put(self.batch)
                self.batch = None

            chained = self._chained
            self._chained = None

        if chained is not None:
            chained.prior_ready()

class _InflightLoad(Generic[D]):
    __slots__ = ("chain", "memory", "_remaining")

    def __init__(self, chain: _LoaderChain[D], memory: _BatchMemory):
        self.chain = chain
        self.memory = memory
        self._remaining = chain.group.data._batch_size

    def _check_done(self) -> bool:
        self._remaining -= 1
        if self._remaining > 0:
            return False

        assert self._remaining == 0
        return True

    def load_succeeded(self, idx: int, result: dict[str, Any]) -> None:
        if self.chain.batch is not None:
            self.chain.batch._item_loaded(idx, result)

        if not self._check_done():
            return

        batch = self.chain.batch
        if batch is None:
            self.memory.recycle()
            return

        layouts = _nonnull(self.chain.group.data._loader_layouts)

        try:
            arrays = self.chain.dataloader._place(
                self.memory.make_arrays(layouts),
                layouts,
                batch._additional,
                self.chain.group.data
            )
        except Exception:
            self.memory.release()
            self.chain.batch_failed()

            traceback.print_exc()
            return

        batch._arrays_loaded(arrays)
        self.chain.batch_ready()

        if isinstance(self.memory, _CacheBatchMemory):
            self.chain.dataloader._cache_save(batch, self.memory)

    def load_failed(self, _idx: int) -> None:
        if self.chain.batch is not None:
            if isinstance(self.memory, _CacheBatchMemory):
                self.chain.dataloader._end_caching(self.chain.batch)

            self.chain.batch_failed()

        if self._check_done():
            self.memory.recycle()

def _default_placement(
    arrays: MappingProxyType[str, np.ndarray],
    layouts: MappingProxyType[str, ArrayLayout],
    _additional: MappingProxyType[str, Sequence[Any]],
    _group: DataGroup[D],
) -> dict[str, jax.Array]:
    return { name: jnp.asarray(array, dtype=layouts[name][2]) for name, array in arrays.items() }

@final
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
    :param depth: The maximum number of batches that can exist at any point in time. Memory usage is proportional to the
        size of the largest possible batch multiplied by the loader depth. This should be at least two (one batch being
        processed, one batch loading), but should be larger if the dataloader needs to work ahead further to amortize
        loading time outliers. The default is 3, allowing the dataloader to work one batch ahead.
    :param loader_count: The number of loader processes. Each loader process loads a single item at a time. This
        defaults to 1, but should probably be higher. Optimally, it should be tuned to saturate the available
        throughput of your data origin (disk/network) without introducing unnecessary process context switching. This
        may be 0 only if all data is exclusively loaded from cache, i.e. no :class:`DataGroup` ``loader_arrays`` are
        specified.
    :param loader_nice: An optional niceness applied to loader processes. Ignored if unsupported by the underlying
        operating system.
    :param cacher_count: The number of cache writer threads. Each cache writer saves a single cache entry at a time.
        This defaults to 1, but should probably be higher. This may be 0 only if all :class:`DataGroup` caches are
        readonly.
    :param start_at: A tuple specifying how far to skip ahead before loading, for example to resume from a checkpoint.
        The first element is the epoch to skip to, and the second is a number of additional batches to skip. The number
        of batches to skip can exceed the number of batches in an epoch, in which case additional epochs are skipped.
        The default is ``(0, 0)``, indicating to start at the beginning.
    :param end_at: An optional tuple specifying when to stop. The first element is either ``"batch"`` or ``"epoch"``,
        and the second specifies which zero-indexed batch or epoch to stop before. So ``("epoch", 1)`` stops after epoch
        0. This argument specifies an absolute position and is not relative to ``start_at``. If this is not specified,
        the dataloader runs until it is interrupted by :func:`interrupt` or its controlling ``with`` block is exited.
    :param interleave_groups: If multiple training DataGroups are specified, this controls how batches from the
        different groups are interleaved within an epoch. If ``False``, groups are loaded sequentially in the order
        specified. If ``True``, the default, batches from different groups are interleaved, with the least-utilized
        earliest-index group being selected for each batch.
    :param timeout_sec: Raise :class:`BatchTimeoutError` if no batches have completed within the specified timeout.
        The default is ``60``, and ``0`` or less disables. A value less than ``20`` is not recommended.
    :param startup_func: An optional callable which is called by each loader process once at startup immediately before
        loading commences. This callable cannot be a lambda, as it must be loadable from a child process.
    :param placement_func: If specified this function is called with a dictionary of NumPy arrays and is responsible for
        orchestrating the placement of the arrays on JAX devices. In addition to the arrays, it is provided their
        layouts, additional data, and the associated :class:`DataGroup`. See
        `<https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement>`_ for details on data placement in JAX. The
        default implementation places batches uncommitted on the default JAX device.

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
                "array_name": ((dim, ...), numpy_dtype, jax_dtype),
                ...
            }

            train = DataGroup(batch_size, my_data[1000:], loader_arrays=array_defs)
            valid = DataGroup(batch_size, my_data[:1000], loader_arrays=array_defs)

            dataloader = Dataloader(
                my_loader,
                train,
                validation = ("epoch", 1, valid), # run validation after every epoch
                end_at = ("epoch", 5)             # run 5 epochs in total
            )

            with dataloader: # a with block is required
                # consider using hydrax.tqdm.tbatches instead of a vanilla for loop here
                for batch in dataloader:
                    if isinstance(batch, TrainingBatch):
                        run_training_batch(batch)
                    elif isinstance(batch, ValidationBatch):
                        run_validation_batch(batch)

                    del batch # important, release batch before waiting for next one or cleaning up

    .. important::
        Read the documentation for your ``loader_func`` carefully. If you receive a warning from Hydrax about your
        loader, you should fix your code. Failure to do this could result in your batch data changing out from
        underneath you, leading to significant training issues such as NaNs.

        If you're using Python's built in ``for`` loop to iterate over batches, it's important to remember not to
        accidentally retain a reference to a batch while "going around" the loop. Python's local variables are not
        scoped. See the code example above for a way to address this with ``del``.

        If you are experiencing deadlocks as a result of retaining batch or array references between iterations,
        consider using :func:`debug_batch_references` or
        `gc.get_referrers <https://docs.python.org/3/library/gc.html#gc.get_referrers>`_ to find out what's holding on
        to your batches, though do keep in mind that JAX dispatch will retain references while running ahead. You can
        check your work by running the Dataloader with ``depth = 1``, which will immediately deadlock if the first batch
        is not properly released.

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
        "_loader", "_depth", "_startup", "_loader_count", "_loader_nice", "_start_at", "_end_at", "_interleave",
        "_timeout", "_tgroups", "_buffer_size", "_vmode", "_vinterval", "_vgroups", "_vepoch", "_vbatch",
        "_batches_per_validation", "_interrupt", "_abort", "_failed", "_setup", "_running", "_watchdog_armed",
        "_idle_usec", "_first_batch", "_tgroup", "_mpctx", "_batches", "_semaphore", "_buffers", "_memories",
        "_submission_id", "_submission_queue", "_completion_queue", "_inflight", "_submission_thread",
        "_completion_thread", "_watchdog_thread", "_loaders", "_chain_lock", "_chain_to", "_batches_per_epoch",
        "_sigint", "_placement", "_cacher_count", "_cacher_queue", "_cachers", "_cache_inflight",
        "_cache_inflight_lock", "_cacher_semaphore"
    )

    def __init__(
        self,
        loader_func: LoaderFunc,
        training: Iterable[DataGroup[D]] | DataGroup[D],
        *,
        validation: tuple[str, int, Iterable[DataGroup[D]] | DataGroup[D]] | None = None, # ("epoch" | "batch"), interval, data
        depth: int = 3,
        loader_count: int = 1,
        loader_nice: int = 0,
        cacher_count: int = 1,
        start_at: tuple[int, int] = (0, 0), # epoch, epoch batch
        end_at: tuple[str, int] | None = None, # ("never" | "epoch" | "batch"), interval
        interleave_groups: bool = True,
        timeout_sec: int = 60,
        startup_func: StartupFunc | None = None,
        placement_func: PlacementFunc | None = None
    ):
        if type(self) is not Dataloader: # pylint: disable=unidiomatic-typecheck
            warn(
                f"{type(self).__name__} derives from hydrax.Dataloader. This is not supported.",
                SyntaxWarning
            )

        if is_worker():
            raise RuntimeError("Dataloader cannot run in a worker process.")

        self._loader = loader_func
        self._depth = depth
        self._loader_count = loader_count
        self._cacher_count = cacher_count
        self._startup = startup_func
        self._loader_nice = loader_nice
        self._end_at = end_at if end_at is not None else ("never", -1)
        self._interleave = interleave_groups
        self._timeout = timeout_sec if timeout_sec > 0 else 31536000 # not technically disabled, but this is 1 year.
        self._placement = placement_func if placement_func is not None else _default_placement

        if self._end_at[0] not in ('never', 'epoch', 'batch'):
            raise ValueError("end_at endpoint must be 'never', 'epoch', or 'batch'")

        self._tgroups = _as_groups(self, training, 0)

        self._batches_per_epoch = 0
        self._buffer_size = 0
        needs_loaders = False
        needs_cachers = False

        for tgroup in self._tgroups:
            tgroup.data._prepare()

            self._batches_per_epoch += tgroup.data.batch_count

            if tgroup.data._cache_direct:
                needs_loaders = True
            elif tgroup.data._loader_size is not None:
                needs_loaders = True
                self._buffer_size = max(self._buffer_size, tgroup.data._loader_size)

            if not tgroup.data._cache_readonly:
                needs_cachers = True

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
            self._vgroups = _as_groups(self, validation[2], len(self._tgroups))
            self._vepoch = -1
            self._vbatch = -1
            self._batches_per_validation = 0

            for vgroup in self._vgroups:
                vgroup.data._prepare()

                self._batches_per_validation += vgroup.data.batch_count

                if vgroup.data._cache_direct:
                    needs_loaders = True
                elif vgroup.data._loader_size is not None:
                    needs_loaders = True
                    self._buffer_size = max(self._buffer_size, vgroup.data._loader_size)

                if not vgroup.data._cache_readonly:
                    needs_cachers = True
        else:
            self._vmode = "none"
            self._batches_per_validation = 0

        if needs_loaders:
            if self._loader_count < 1:
                raise ValueError("loaders are required, cannot provision none")

            if self._loader_count > self._depth:
                warn("ignoring loader_count ({self._loader_count}) in excess of depth ({self._depth})", ResourceWarning)
                self._loader_count = self._depth
        else:
            print("[hydrax] loading exclusively from cache")
            self._loader_count = 0

        if needs_cachers:
            if self._cacher_count < 1:
                raise ValueError("cachers are required, cannot provision none")
            if self._cacher_count > self._depth:
                warn("ignoring cacher_count ({self._cacher_count}) in excess of depth ({self._depth})", ResourceWarning)
                self._cacher_count = self._depth
        else:
            self._cacher_count = 0

        self._interrupt = False
        self._abort = False
        self._failed = False

        self._setup = False
        self._running = False
        self._watchdog_armed = False

        self._idle_usec = 0
        self._first_batch = True
        self._sigint: Any = None

        self._semaphore = BoundedSemaphore(depth)
        self._batches = queue.SimpleQueue() # type: queue.SimpleQueue[Batch[D] | None]
        self._buffers: list[SharedMemory] = []
        self._memories: list[SharedMemory] = []

        if self._loader_count > 0:
            self._mpctx = mp.get_context("spawn")
            self._submission_id = 0
            self._submission_queue = self._mpctx.SimpleQueue() # type: mp.SimpleQueue[_Submission | None]
            self._completion_queue = self._mpctx.SimpleQueue() # type: mp.SimpleQueue[_Completion | None]
            self._inflight: dict[int, tuple[_InflightLoad, int]] = {}

        if self._cacher_count > 0:
            self._cacher_queue = queue.SimpleQueue() # type: queue.SimpleQueue[_CacheSave | None]
            self._cacher_semaphore = BoundedSemaphore(depth)
            self._cache_inflight: set[str] = set()
            self._cache_inflight_lock = Lock()

        self._submission_thread: Thread | None = None
        self._completion_thread: Thread | None = None
        self._watchdog_thread: Thread | None = None
        self._loaders: list[Process] = []
        self._cachers: list[Thread] = []

        self._chain_to: _LoaderChain | None = None
        self._chain_lock = Lock()

    def __enter__(self) -> "Dataloader[D]":
        """Use via a ``with`` block."""

        if self._setup or self._failed:
            raise RuntimeError("dataloader is already set up")

        self._setup = True
        self._running = True
        self._interrupt = False

        self._sigint = signal(SIGINT, self._handle_sigint)
        self._idle_usec = 0
        self._first_batch = True

        if not self._interleave:
            self._tgroup = 0

        if self._vmode != "none":
            self._vepoch = 0
            self._vbatch = 0

        if self._buffer_size > 0:
            for _ in range(self._depth):
                memory = SharedMemory(create=True, size=self._buffer_size)
                self._memories.append(memory)
                self._buffers.append(memory)

        if self._loader_count > 0:
            tlayouts = [group.data._loader_layouts for group in self._tgroups]
            vlayouts = [group.data._loader_layouts for group in self._vgroups] if self._vmode != "none" else []

            loader_args = (
                self._loader,
                self._startup,
                self._loader_nice,
                self._submission_queue,
                self._completion_queue,
                [memory.name for memory in self._memories],
                tlayouts + vlayouts
            )

            for i in range(self._loader_count):
                loader = self._mpctx.Process(
                    target=_run_loader,
                    name=f"hydrax-loader-{i}",
                    args=loader_args,
                    daemon=True
                )
                loader.start()

                self._loaders.append(cast(Process, loader))

            self._watchdog_armed = True
            self._watchdog_thread = Thread(
                target=self._run_watchdog,
                name="hydrax-watchdog",
                daemon=True
            )
            self._watchdog_thread.start()

            self._completion_thread = Thread(
                target=self._run_completion,
                name="hydrax-completion",
                daemon=True
            )
            self._completion_thread.start()

        if self._cacher_count > 0:
            for i in range(self._cacher_count):
                cacher = Thread(
                    target=self._run_cacher,
                    name=f"hydrax-cacher-{i}",
                    daemon=True
                )
                cacher.start()

                self._cachers.append(cacher)

        self._submission_thread = Thread(
            target=self._run_submission,
            name="hydrax-submission",
            daemon=True
        )
        self._submission_thread.start()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        if not self._setup:
            raise RuntimeError("dataloader is not set up")

        if self._failed:
            self._setup = False
            self._running = False
            self._restore_sigint()

            print(f"""\
[hydrax]: dataloader failed
[hydrax]:   semaphore: {self._semaphore}
[hydrax]:   buffers: {len(self._buffers) if hasattr(self, "_buffers") else "N/A"}
[hydrax]:   inflight: {self._inflight if hasattr(self, "_inflight") else "N/A"}
[hydrax]:   cache semaphore: {self._cacher_semaphore if hasattr(self, "_cacher_semaphore") else "N/A"}
[hydrax]:   cache inflight: {self._cache_inflight if hasattr(self, "_cache_inflight") else "N/A"}
""")
            return

        if exc_type is not None:
            print(f"[hydrax]: shutting down due to {exc_type.__name__}")
            self._timeout = 20

        if self._running:
            self._interrupt = True

            try:
                while self._get_batch() is not None:
                    pass
            except Exception:
                self._failed = True
                self._setup = False
                self._running = False
                self._restore_sigint()
                return

            self._running = False

        assert self._submission_thread is not None
        self._submission_thread.join()
        self._submission_thread = None

        if self._cacher_count > 0:
            for _ in self._cachers:
                self._cacher_queue.put(None)

            for cacher in self._cachers:
                cacher.join()

            self._cachers.clear()

        gc.collect()

        for memory in self._memories:
            memory.close()
            memory.unlink()

        self._buffers.clear()
        self._memories.clear()

        self._setup = False
        self._restore_sigint()

    def _restore_sigint(self) -> None:
        signal(SIGINT, self._sigint)
        self._sigint = None

    def __iter__(self) -> "Dataloader[D]":
        if not self._running:
            raise RuntimeError("dataloader is not running")

        return self

    def __next__(self) -> Batch[D]:
        """Retrieves the next :class:`Batch`."""

        if not self._running or self._failed:
            raise RuntimeError("dataloader is not running")

        try:
            batch = self._batches.get_nowait()
            has_batch = True
        except queue.Empty:
            has_batch = False

        if not has_batch:
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
        """Returns the total amount of time, in microseconds, since the last call to ``idle_usec``, that
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

        if not self._setup:
            raise RuntimeError("dataloader is not set up")

        self._interrupt = True

    def _place(
        self,
        arrays: dict[str, np.ndarray],
        layouts: dict[str, ArrayLayout],
        additional: dict[str, list[object]],
        group: DataGroup[D],
    ) -> dict[str, jax.Array]:
        arrays = self._placement(
            MappingProxyType(arrays),
            MappingProxyType(layouts),
            MappingProxyType(additional),
            group
        )

        if self._placement is not _default_placement:
            _check_jax_arrays(arrays, layouts)

        return arrays

    def _submit_to_loaders(self, inflight: _InflightLoad[D], batch_idx: int, data_idx: int, seed: int | None) -> None:
        sid = self._submission_id
        self._submission_id += 1

        self._inflight[sid] = (inflight, batch_idx)

        try:
            self._submission_queue.put((
                sid,
                inflight.chain.group.data._data[data_idx],
                inflight.memory.name,
                inflight.chain.group.groupid,
                batch_idx,
                seed
            ))
        except Exception:
            inflight.load_failed(batch_idx)
            del self._inflight[sid]

            traceback.print_exc()

    def _load_from_cache(self, chain: _LoaderChain, batch: Batch[D]) -> bool:
        if chain.group.data._cache_location is None:
            return False

        layouts = _nonnull(chain.group.data._cache_layouts)

        try:
            memory = _CacheBatchMemory.open(self, batch)
        except FileNotFoundError:
            return False
        except Exception:
            traceback.print_exc()
            return False

        try:
            arrays = memory.make_arrays(layouts)

            memory.buffer.seek(_nonnull(chain.group.data._cache_size))
            additional = pickle.load(memory.buffer)
        except Exception:
            memory.abandon()

            traceback.print_exc()
            return False

        try:
            arrays = self._place(arrays, layouts, additional, chain.group.data)
        except Exception:
            memory.abandon()

            chain.batch_failed()
            self._semaphore.release()

            traceback.print_exc()
            return True

        batch._from_cached(arrays, additional)
        chain.batch_ready()
        return True

    def _load_into_cache(self, chain: _LoaderChain, batch: Batch[D]) -> bool:
        if not chain.group.data._cache_direct:
            return False

        if not self._begin_caching(batch):
            return False

        try:
            memory = _CacheBatchMemory.create(self, batch)
        except Exception:
            chain.batch_failed()
            self._end_caching(batch)
            self._semaphore.release()

            traceback.print_exc()
            return True

        batch._submit_to_loaders(_InflightLoad(chain, memory))
        return True

    def _load_from_worker(self, chain: _LoaderChain, batch: Batch[D]) -> bool:
        if chain.group.data._loader_layouts is None:
            return False

        batch._submit_to_loaders(_InflightLoad(chain, _SharedBatchMemory(self, self._buffers.pop())))
        return True

    def _begin_caching(self, batch: Batch[D]) -> bool:
        with self._cache_inflight_lock:
            if batch._cache_path in self._cache_inflight:
                return False

            self._cache_inflight.add(_nonnull(batch._cache_path))

        return True

    def _end_caching(self, batch: Batch[D]):
        self._cache_inflight.remove(_nonnull(batch._cache_path))

    def _cache_save(self, batch: Batch[D], memory: _CacheBatchMemory | None) -> None:
        self._cacher_semaphore.acquire() # pylint: disable=consider-using-with
        self._cacher_queue.put((batch, memory))

    def _run_submission(self) -> None:
        self._submit_batches()

        self._chain_to = None

        if self._loader_count > 0:
            self._watchdog_armed = False

            for _ in self._loaders:
                self._submission_queue.put(None)

            for loader in self._loaders:
                loader.join()

            self._loaders.clear()

            assert self._completion_thread is not None
            self._completion_queue.put(None)
            self._completion_thread.join()
            self._completion_thread = None

            assert self._watchdog_thread is not None
            self._watchdog_thread.join()
            self._watchdog_thread = None

        gc.collect()

        tries = 0
        for _ in range(self._depth):
            while not self._semaphore.acquire(timeout=1): # pylint: disable=consider-using-with
                gc.collect()

                tries += 1
                if tries == 5 or (tries > 5 and (tries - 5) % 15 == 0):
                    debug_batch_references()

        self._batches.put(None)

    def _submit_batches(self) -> None:
        (epoch, seek) = self._start_at
        batch = epoch * self._batches_per_epoch

        for tgroup in self._tgroups:
            tgroup.prepare(epoch)

        if self._vmode != "none":
            for vgroup in self._vgroups:
                vgroup.prepare(epoch)

        while True:
            ebatch = 0

            for tgroup in self._tgroups:
                tgroup.shuffle(epoch)

            while (group := self._select_group()) is not None:
                while (reservation := group.reserve_training(epoch)) is not None:
                    if seek == 0:
                        if not self._chain_load(
                            TrainingBatch(
                                group,
                                reservation[0],
                                reservation[1],
                                epoch,
                                ebatch,
                                batch,
                                reservation[2]
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

        for vgroup in self._vgroups:
            while (reservation := vgroup.reserve_validation()) is not None:
                if not self._chain_load(
                    ValidationBatch(
                        vgroup,
                        reservation[0],
                        reservation[1],
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

    def _chain_load(self, batch: Batch[D]) -> bool:
        while True:
            acquired = self._semaphore.acquire(timeout=1) # pylint: disable=consider-using-with

            if self._interrupt:
                if acquired:
                    self._semaphore.release()

                return False

            if acquired:
                break

        chain = _LoaderChain(self, batch)

        if self._chain_to is None:
            chain.prior_ready()
        else:
            self._chain_to.chain_to(chain)

        self._chain_to = chain

        if self._load_from_cache(chain, batch):
            return True

        if self._load_into_cache(chain, batch):
            return True

        if self._load_from_worker(chain, batch):
            return True

        chain.batch_failed()
        self._semaphore.release()
        return True

    def _run_completion(self) -> None:
        while (completion := self._completion_queue.get()) is not None:
            sid, result = completion
            inflight, batch_idx = self._inflight.pop(sid)

            if result is None:
                inflight.load_failed(batch_idx)
            else:
                inflight.load_succeeded(batch_idx, result)

            del result, inflight

    def _run_watchdog(self) -> None:
        mp.connection.wait([loader.sentinel for loader in self._loaders])

        if self._watchdog_armed:
            print("[hydrax] loader process exited, aborting")
            self._abort = True

    def _run_cacher(self) -> None:
        while (cachesave := self._cacher_queue.get()) is not None:
            batch, memory = cachesave

            try:
                if not self._interrupt:
                    if memory is None:
                        self._cache_save_arrays(batch)
                    else:
                        self._cache_save_direct(batch, memory)
                elif memory is not None:
                    try:
                        os.remove(memory.name)
                    except SystemError:
                        pass
            except Exception:
                traceback.print_exc()
            finally:
                self._end_caching(batch)
                self._cacher_semaphore.release()

            del batch, memory, cachesave

    def _cache_save_arrays(self, batch: Batch[D]) -> None:
        path = _nonnull(batch._cache_path)
        tmp = f"{path}.tmp"
        layouts = _nonnull(batch._group.data._cache_layouts)

        try:
            with open(tmp, mode="wb", buffering=0) as file:
                for (name, (_shape, np_dtype, _jax_dtype, offset, _count)) in layouts.items():
                    file.seek(offset)
                    np.asarray(batch._arrays[name], dtype=np_dtype).tofile(file)

                file.seek(_nonnull(batch._group.data._cache_size))
                pickle.dump(batch._additional, file)

                file.flush()
                os.fsync(file.fileno())

            os.rename(tmp, path)
        except:
            try:
                os.remove(tmp)
            except SystemError:
                pass

            raise

    def _cache_save_direct(self, batch: Batch[D], memory: _CacheBatchMemory) -> None:
        try:
            memory.buffer.flush()

            with open(memory.take_fd(), mode="ab", buffering=4096) as file:
                pickle.dump(batch._additional, file)
                file.flush()
                os.fsync(file.fileno())

            os.rename(memory.name, _nonnull(batch._cache_path))
        except:
            try:
                os.remove(memory.name)
            except SystemError:
                pass

            raise

    def _handle_sigint(self, _signum, _frame) -> None:
        if self._abort:
            print("[hydrax] KeyboardInterrupt: terminating")
            os._exit(1)
        elif self._interrupt:
            print("[hydrax] KeyboardInterrupt: aborting, repeat to terminate")
            self._watchdog_armed = False
            self._abort = True
        else:
            print("[hydrax] KeyboardInterrupt: interrupting, repeat to abort")
            self._interrupt = True

def _trace_references(obj: tuple[object], ignore: set[int], visited: set[int], depth: int) -> None:
    if depth >= 12:
        print(f"[hydrax]   {'  ' * depth}⮤ ...")
        return

    referrers = gc.get_referrers(obj[0])
    ignore.add(id(obj))
    ignore.add(id(referrers))

    while len(referrers) > 0:
        ref = referrers.pop()
        if id(ref) in ignore:
            continue

        try:
            rep = repr(ref)
        except Exception:
            rep = ""

        if len(rep) == 0 or len(rep) > 80:
            ty = type(ref)

            mod = ty.__module__
            if mod is None:
                mod = ""
            else:
                mod += "."

            rep = f"<{mod}{ty.__name__} object at 0x{id(ref):x}>"

        print(f"[hydrax]   {'  ' * depth}⮤ {rep}")

        if id(ref) in visited:
            print(f"[hydrax]     {'  ' * depth}⮤ ...")
            continue

        visited.add(id(ref))

        tup = (ref,)
        del ref

        _trace_references(tup, ignore, visited, depth + 1)

    ignore.remove(id(referrers))
    ignore.remove(id(obj))

def debug_batch_references() -> None:
    """Prints a partial tree of all traceable references to batch memory."""

    objects = gc.get_objects()
    visited: set[int] = set()
    ignore: set[int] = set([id(objects)])

    while len(objects) > 0:
        obj = objects.pop()

        if not isinstance(obj, _BatchMemory) or obj.recycled:
            continue

        print(f"[hydrax] remaining references to {obj}:")

        tup = (obj,)
        del obj

        _trace_references(tup, ignore, visited, 0)

def _layout_subarrays(layout: dict[str, ArrayLayout] | None) -> list[_SubarrayLayout] | None:
    if layout is None:
        return None

    layouts = []

    for (name, (shape, np_dtype, _jax_dtype, offset, count)) in layout.items():
        batch_count = count // shape[0]
        assert batch_count * shape[0] == count

        layouts.append((name, shape[1:], np_dtype, batch_count, offset, batch_count * np_dtype.itemsize))

    return layouts

def _run_loader(
    loader: LoaderFunc,
    startup: StartupFunc | None,
    nice: int,
    submission_queue: "mp.SimpleQueue[_Submission | None]",
    completion_queue: "mp.SimpleQueue[_Completion | None]",
    shm_names: list[str],
    group_layouts: list[dict[str, ArrayLayout] | None],
) -> None:
    signal(SIGINT, SIG_IGN)

    if nice != 0 and hasattr(os, "nice"):
        os.nice(nice)

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if startup is not None:
        startup()

    shms: dict[str, SharedMemory] = {}
    group_subarray_layouts = [_layout_subarrays(layout) for layout in group_layouts]

    try:
        for shm_name in shm_names:
            shms[shm_name] = SharedMemory(shm_name)

        while (work := submission_queue.get()) is not None:
            (sid, data, file_name, group_id, batch_idx, seed) = work

            try:
                shm = shms.get(file_name)
                if shm is None:
                    memory = _CacheLoaderMemories(
                        file_name,
                        _nonnull(group_subarray_layouts[group_id]),
                        batch_idx
                    ) # type: _CacheLoaderMemories | _SharedLoaderMemory
                else:
                    memory = _SharedLoaderMemory(
                        shm,
                        _nonnull(group_layouts[group_id]),
                        batch_idx
                    )

                try:
                    result = loader(data, MappingProxyType(memory.make_arrays()), seed)

                    if result is None:
                        result = {}
                    else:
                        _check_item_additional_data(result)
                finally:
                    memory.freeze()
                    del memory

                completion_queue.put((sid, result))
            except Exception:
                completion_queue.put((sid, None))

                print(f"[hydrax] exception while loading data item: {data}")
                traceback.print_exc()
    finally:
        gc.collect()

        for shm in shms.values():
            shm.close()
