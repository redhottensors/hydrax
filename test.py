from hydrax import Dataloader, DataGroup, Batch
from hydrax.tqdm import tbatches

import numpy as np
import cProfile

from types import MappingProxyType
from typing import Iterable, Dict, Any

def loader(data: int, arrays: MappingProxyType[str, np.ndarray], seed: int | None) -> Dict[str, Any]:
    for i in range(3):
        for j in range(4):
            arrays["array"][i][j] = data + (i * 100 + j)

    return { "my_data_was": data }

if __name__ == "__main__":
    def verify(base: int, batch: Batch) -> None:
        assert(batch.additional["my_data_was"] == [base, base + 1000])

        for b in range(2):
            for i in range(3):
                for j in range(4):
                    assert(batch.arrays["array"][b][i][j] == base + b * 1000 + i * 100 + j)

    def main() -> None:
        data = range(1000, 10000, 1000)
        group = DataGroup(
            batch_size = 2,
            arrays = { "array": ((3, 4), np.dtype("int32")) },
            data = data,
        )

        dl = Dataloader(
            loader_func = loader,
            training = group,
            loader_count = 2,
            shuffle_groups = "none",
            end_at = ("epoch", 100),
            timeout_sec = 10
        )

        it = iter(tbatches(dl, report_interval = 9))

        for _ in range(100):
            verify(1000, next(it))
            verify(3000, next(it))
            verify(5000, next(it))
            verify(7000, next(it))

        assert(next(it, None) is None)

    cProfile.run("main()", "test.prof")
