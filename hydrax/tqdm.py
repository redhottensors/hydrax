"""Provides `tqdm <https://github.com/tqdm/tqdm>`_ progress bar support."""

from tqdm import tqdm
from hydrax import Dataloader, Batch, TrainingBatch, ValidationBatch
from typing import Iterable, Dict, Any, Self, Generic, TypeVar

D = TypeVar('D')

class ProgressMonitor(Generic[D]):
    """Wraps a :class:`hydrax.Dataloader` with `tqdm <https://github.com/tqdm/tqdm>`_ progress bars.

    .. tip::
        In most cases, you can use :func:`tbatches` instead, which wraps a dataloader and yields its batches.

    :param dataloader: The :class:`hydrax.Dataloader` to wrap.
    :param report_interval: Interval, in batches, to check for issues and report. 0 by default, which means reporting
        is disabled.
    :param description: A description for the progress bar. Defaults to "train".
    :param kwargs: Additional keyword arguments passed to tqdm.
    """

    __slots__ = (
        "_dataloader", "_tqdm_args", "_report", "_tbar", "_vbar", "_prev_tbatch", "_prev_vbatch", "_count", "_dropped"
    )

    def __init__(
        self,
        dataloader: Dataloader[D],
        report_interval: int = 0,
        description: str = "train",
        **kwargs
    ):
        self._dataloader = dataloader
        self._tqdm_args = kwargs
        self._report = report_interval

        self._tbar: tqdm | None = tqdm(
            initial=dataloader.first_batch,
            total=dataloader.last_batch,
            unit="batch",
            desc=description,
            leave=False,
            **kwargs
        )

        self._vbar: tqdm | None = None
        self._prev_tbatch = dataloader.first_batch - 1
        self._prev_vbatch = -1
        self._count = 0
        self._dropped = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:
        self.close()

    def begin_batch(self, batch: Batch[D]) -> None:
        """Signal the start of a batch."""

        assert(self._tbar is not None)

        if isinstance(batch, ValidationBatch):
            if self._vbar is None:
                self._vbar = tqdm(
                    total=self._dataloader.batches_per_validation,
                    unit="batch",
                    desc="validation",
                    leave=False,
                    **self._tqdm_args
                )
                self._prev_vbatch = -1
        elif self._vbar is not None:
            self._vbar.close()
            self._vbar = None
            self._tbar.unpause()

    def end_batch(self, batch: Batch[D]) -> None:
        """Signal the end of a batch."""

        assert(self._tbar is not None)

        if isinstance(batch, TrainingBatch):
            assert(self._vbar is None)

            delta = batch.batch_num - self._prev_tbatch
            assert(delta > 0)

            self._tbar.update(delta)
            self._dropped += delta - 1
            self._prev_tbatch = batch.batch_num

        if isinstance(batch, ValidationBatch):
            assert(self._vbar is not None)

            delta = batch.validation_epoch_batch - self._prev_vbatch
            assert(delta > 0)

            self._vbar.update(delta)
            self._dropped += delta - 1
            self._prev_vbatch = batch.validation_epoch_batch

        self._count += 1

        if self._report > 0 and self._count % self._report == 0:
            idle_ms = int((self._dataloader.idle_usec() / self._report) / 1000)

            if idle_ms > 0:
                print(f"[hydrax]: running behind by {idle_ms:.0}ms/batch")

            if self._dropped > 0:
                pct = (self._dropped * 100) / (self._dropped + self._report)
                print(f"[hydrax]: dropped {self._dropped} {'batch' if self._dropped == 1 else 'batches'} ({pct:.0}%)")
                self._dropped = 0

    def close(self) -> None:
        """Signal the end of monitoring."""

        if self._vbar is not None:
            self._vbar.close()
            self._vbar = None

        if self._tbar is not None:
            self._tbar.close()
            self._tbar = None

def tbatches(
    dataloader: Dataloader[D],
    report_interval: int = 0,
    description: str = "train",
    **kwargs
) -> Iterable[Batch[D]]:
    """Wraps a :class:`hydrax.Dataloader` with `tqdm <https://github.com/tqdm/tqdm>`_ progress bars, yielding each
    :class:`hydrax.Batch`.

    :param dataloader: The :class:`hydrax.Dataloader` to wrap.
    :param report_interval: Interval, in batches, to check for issues and report. 0 by default, which means reporting
        is disabled.
    :param desc: A description for the progress bar. Defaults to "train".
    :param kwargs: Additional keyword arguments passed to tqdm.
    :return: An iterator over each :class:`hydrax.Batch` produced by the Dataloader.
    """

    with ProgressMonitor(dataloader, report_interval, description, **kwargs) as monitor:
        with dataloader:
            for batch in dataloader:
                monitor.begin_batch(batch)
                yield batch
                monitor.end_batch(batch)
