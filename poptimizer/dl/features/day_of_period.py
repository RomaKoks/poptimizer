"""Номер дня внутри исторического периода."""
from typing import Tuple

import torch

from poptimizer.dl.features.data_params import DataParams
from poptimizer.dl.features.feature import Feature, FeatureType


class DayOfPeriod(Feature):
    """Номер для в историческом периоде.

    Сверточные сети инвариантны к сдвигу, но для построения прогноза разные дни исторического периода
    могут иметь разную важность.
    """

    def __init__(self, ticker: str, params: DataParams):
        super().__init__(ticker, params)

        self.history_days = params.history_days

    def __getitem__(self, item: int) -> torch.Tensor:
        return torch.arange(self.history_days)

    @property
    def type_and_size(self) -> Tuple[FeatureType, int]:
        """Тип признака и размер признака."""
        return FeatureType.EMBEDDING_SEQUENCE, self.history_days
