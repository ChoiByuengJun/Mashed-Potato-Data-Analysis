from abc import ABC, abstractmethod
from pandas import DataFrame

class FeatureEngineeringService(ABC):
    @abstractmethod
    def featureEngineering(self, file_path: str) -> (DataFrame, dict):
        pass
