from abc import ABC, abstractmethod
from pandas import DataFrame

class FeatureEngineeringService(ABC):
    @abstractmethod
    def feature_engineering(self, file_path: str) -> (DataFrame, dict):
        pass
