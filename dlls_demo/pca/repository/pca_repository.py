from abc import ABC, abstractmethod
import pandas as pd

class PCARepository(ABC):
    @abstractmethod
    def loadData(self, file_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def scaleData(self, data: pd.DataFrame):
        pass

    @abstractmethod
    def applyPCA(self, scaled_data, data_columns, n_components: int):
        pass

    @abstractmethod
    def createHeatmap(self, components, features):
        pass
