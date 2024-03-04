"""Provides support for using `Pandas <https://pandas.pydata.org/docs/>`_ DataFrames in a :class:`hydrax.DataGroup`."""

from pandas import DataFrame, Series # type: ignore[import-untyped]

from typing import Sequence

class RowData(Sequence[Series]):
    """Wraps a `Pandas <https://pandas.pydata.org/docs/>`_ ``DataFrame`` as a row-wise sequence of ``Series``."""

    __slots__ = ("_dataframe")

    def __init__(self, dataframe: DataFrame):
        self._dataframe = dataframe

    @property
    def dataframe(self) -> DataFrame:
        """The backing Pandas DataFrame."""
        return self._dataframe

    def __len__(self) -> int:
        return len(self._dataframe.index)

    def __getitem__(self, key: int | slice) -> "RowData | Series":
        if isinstance(key, int):
            return self._dataframe[key]

        if isinstance(key, slice):
            return RowData(self._dataframe[key])

        raise TypeError(f"invalid key type: {type(key).__name__}")
