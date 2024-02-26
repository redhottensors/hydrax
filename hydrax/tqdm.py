"""Provides :func:`tbatches`, which wraps a :class:`hydrax.Dataloader` with tqdm progress bars."""

from tqdm import tqdm
from hydrax import Dataloader
from typing import Iterable, Dict, Any

def tbatches(dl: Dataloader, report_interval: int = 0, desc: str = "train", **kwargs) -> Iterable[Dict[str, Any]]:
    """Wraps a :class:`hydrax.Dataloader` with tqdm progress bars.

    The Dataloader must have been started via a ``with`` block.

    :param report_interval: Interval, in batches, to check for issues and report. 0 by default, which means reporting is disabled.
    :param desc: A description for the progress bar. Defaults to "train".
    :param kwargs: Additional keyword arguments passed to tqdm.
    :return: An iterator over batches.
    """

    vbar = None
    prev_tbatch = dl.first_batch - 1
    prev_vbatch = -1
    count = 0
    dropped = 0

    with tqdm(
        total=dl.last_batch,
        initial=dl.first_batch,
        unit="batch", desc=desc,
        **kwargs
    ) as tbar:
        try:
            for batch in dl:
                if batch['_validation']:
                    if vbar is None:
                        vbar = tqdm(total=dl.batches_per_validation, unit="batch", desc="validation", leave=False, **kwargs)
                        prev_vbatch = -1
                elif vbar is not None:
                    vbar.close()
                    vbar = None
                    tbar.unpause()

                yield batch

                if vbar is None:
                    tbatch = batch['_batch']
                    tbar.update(tbatch - prev_tbatch)
                    dropped += (tbatch - (prev_tbatch + 1))
                    prev_tbatch = tbatch
                else:
                    vbatch = batch['_validation_epoch_batch']
                    vbar.update(vbatch - prev_vbatch)
                    dropped += (vbatch - (prev_vbatch + 1))
                    prev_vbatch = vbatch

                count += 1
                if report_interval > 0 and count % report_interval == 0:
                    idle_ms = int((dl.idle_usec() / report_interval) / 1000)

                    if idle_ms > 0:
                        print(f"[hydrax]: running behind by {idle_ms:.0}ms/batch")

                    if dropped > 0:
                        pct = (dropped * 100) / (dropped + report_interval)
                        print(f"[hydrax]: dropped {dropped} {'batch' if dropped == 1 else 'batches'} ({pct:.0}%)")
                        dropped = 0
        finally:
            if vbar is not None:
                vbar.close()
                vbar = None
