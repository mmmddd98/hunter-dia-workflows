from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd


class SearchResultReader(ABC):
    @abstractmethod
    def read(self, path: Path) -> pd.DataFrame:
        """Read one search-result file into canonical column names."""
