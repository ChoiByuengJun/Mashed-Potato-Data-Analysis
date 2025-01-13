from abc import ABC, abstractmethod
import pandas as pd

class KMeansRepository(ABC):

    @abstractmethod
    def loadData(self, file_path: str) -> pd.DataFrame:
        pass

    @abstractmethod
    def preprocessData(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        pass

    @abstractmethod
    def scaleData(self, data: pd.DataFrame) -> (pd.DataFrame, object):
        pass

    @abstractmethod
    def performKMeans(self, scaled_data: pd.DataFrame, n_clusters: int) -> (object, list):
        pass

    @abstractmethod
    def addClusterLabels(self, data: pd.DataFrame, labels: list, cluster_type: str) -> pd.DataFrame:
        pass
