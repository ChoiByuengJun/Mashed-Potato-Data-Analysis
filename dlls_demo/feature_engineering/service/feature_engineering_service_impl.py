import pandas as pd
from feature_engineering.repository.feature_engineering_repository_impl import FeatureEngineeringRepositoryImpl
from feature_engineering.service.feature_engineering_service import FeatureEngineeringService

class FeatureEngineeringServiceImpl(FeatureEngineeringService):
    def __init__(self):
        self.repository = FeatureEngineeringRepositoryImpl()

    async def feature_engineering(self, file_path: str):
        data = pd.read_csv(file_path)

        data = self.repository.handleMissingValues(data)
        data, encoders = self.repository.encodeCategoricalFeatures(data)
        data = self.repository.createNewFeatures(data)

        X_train, X_test, y_train, y_test = self.repository.splitTrainTestData(data)
        X_train_scaled, X_test_scaled = self.repository.scaleFeatures(X_train, X_test)

        model = self.repository.trainModel(X_train_scaled, y_train)
        mseError, y_prediction = self.repository.evaluateModel(model, X_test_scaled, y_test)

        comparison = self.repository.compareResult(y_test, y_prediction)
        return {
            "mseError": mseError,
            "comparison": comparison
        }
