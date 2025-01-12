import os
import pandas as pd
from feature_engineering.repository.feature_engineering_repository_impl import FeatureEngineeringRepositoryImpl
from feature_engineering.service.feature_engineering_service import FeatureEngineeringService

class FeatureEngineeringServiceImpl(FeatureEngineeringService):
    def __init__(self):
        # Repository 인스턴스 생성
        self.featureEngineeringRepository = FeatureEngineeringRepositoryImpl()
        self.preprocessed_data_path = os.getenv("PROCESSED_DATA_PATH", "resource/preprocessed_data.csv")

    async def feature_engineering(self, file_path: str = None):
        # 기본 데이터 경로 설정
        if file_path is None:
            file_path = os.getenv("RAW_DATA_PATH", "resource/customer_data.csv")
        print(f"Loading data from: {file_path}")

        # 원본 데이터 로드
        data = pd.read_csv(file_path)
        '''''
        # 결측치 처리
        data = self.featureEngineeringRepository.handleMissingValues(data)
        print("Missing values handled.")
        '''''



        # 새로운 피처 생성
        data = self.featureEngineeringRepository.createNewFeatures(data)
        print("New features created.")

        # 전처리된 데이터 저장
        data.to_csv(self.preprocessed_data_path, index=False)
        print(f"Preprocessed data saved to: {self.preprocessed_data_path}")

        # 범주형 데이터 인코딩 (원핫 인코딩)
        data = self.featureEngineeringRepository.encodeCategoricalFeatures(data)
        print("Categorical features encoded.")

        # 데이터 분리 (학습/테스트)
        X_train, X_test, y_train, y_test = self.featureEngineeringRepository.splitTrainTestData(data)
        print("Data split into training and test sets.")


        # 피처 스케일링
        X_train_scaled, X_test_scaled = self.featureEngineeringRepository.scaleFeatures(X_train, X_test)
        print("Features scaled.")


        # 모델 훈련
        model = self.featureEngineeringRepository.trainModel(X_train, y_train)
        print("Model trained.")

        # 모델 평가
        metrics, y_prediction = self.featureEngineeringRepository.evaluateModel(model, X_test, y_test)
        print(f"Model evaluation completed: {metrics}")

        # 실제값 vs 예측값 비교
        comparison = self.featureEngineeringRepository.compareResult(y_test, y_prediction)
        print("Actual vs Predicted comparison created.")

        # 결과 반환
        return {
            "metrics": metrics,
            "comparison": comparison
        }
