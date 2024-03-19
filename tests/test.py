import cProfile
from types import MappingProxyType
from typing import Dict, Any

import numpy as np

from hydrax import Dataloader, DataGroup, Batch
from hydrax.tqdm import tbatches

EPOCHS = 100
SHAPE = (4, 5)

def loader(data: int, arrays: MappingProxyType[str, np.ndarray], _seed: int | None) -> Dict[str, Any]:
    for i in range(SHAPE[0]):
        for j in range(SHAPE[1]):
            arrays["array"][i][j] = data + (i * 100 + j)

    return { "my_data_was": data }

if __name__ == "__main__":
    def verify(base: int, batch: Batch) -> None:
        assert batch.additional["my_data_was"] == [base, base + 1000]

        for b in range(2):
            for i in range(SHAPE[0]):
                for j in range(SHAPE[1]):
                    assert batch.arrays["array"][b][i][j] == base + b * 1000 + i * 100 + j

    def main() -> None:
        data = range(1000, 10000, 1000)
        group = DataGroup(
            2, data,
            loader_arrays = { "array": (SHAPE, "int32", "int32") },
            cache_location = "_test_cache",
            shuffle_later = "repeat"
        )

        dl = Dataloader(
            loader_func = loader,
            training = group,
            loader_count = 2,
            end_at = ("epoch", EPOCHS),
            timeout_sec = 20
        )

        it = iter(tbatches(dl, report_interval = 9))

        for _ in range(EPOCHS):
            verify(1000, next(it))
            verify(3000, next(it))
            verify(5000, next(it))
            verify(7000, next(it))

        assert next(it, None) is None

    cProfile.run("main()", "test.prof")
