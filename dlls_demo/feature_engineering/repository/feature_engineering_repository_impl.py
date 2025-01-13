import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from feature_engineering.repository.feature_engineering_repository import FeatureEngineeringRepository

class FeatureEngineeringRepositoryImpl(FeatureEngineeringRepository):
    '''''
    def handleMissingValues(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column] = data[column].fillna(data[column].mode().iloc[0])
            else:
                data[column] = data[column].fillna(data[column].mean())
        return data
    '''''



    def createNewFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        # datetime으로 변환
        data['가입 일자'] = pd.to_datetime(data['가입 일자'])
        data['최근 서비스 이용 날짜'] = pd.to_datetime(data['최근 서비스 이용 날짜'])
        data['구매 일자'] = pd.to_datetime(data['구매 일자'])

        ## 데이터 정렬 (CustomerID 및 구매 일자 기준)
        data = data.sort_values(by=['CustomerID', '구매 일자']).reset_index(drop=True)

        # 새로운 피처 생성
        data['가입 기간'] = (data['최근 서비스 이용 날짜'] - data['가입 일자']).dt.days
        #data['서비스 공백 기간'] = (pd.Timestamp.now() - data['최근 서비스 이용 날짜']).dt.days
        data['평균 거래 주기'] = (
            data.groupby('CustomerID')['구매 일자']
            .diff()
            .dt.days
            .groupby(data['CustomerID'])
            .transform('mean')
        )
        # 구매 횟수가 1회인 고객 처리
        data.loc[data['구매 횟수'] <= 1, '평균 거래 주기'] = 0

        data['평균 거래 금액'] = data['총 거래 금액'] / (data['구매 횟수'] + 1e-9)
        return data

    def savePreprocessedData(self, data: pd.DataFrame):
        file_path = os.getenv("PREPROCESSED_DATA_PATH", "resource/preprocessed_data.csv")
        data.to_csv(file_path, index=False)
        print(f"Preprocessed data saved to {file_path}")

    def encodeCategoricalFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        categorical_columns = data.select_dtypes(include=['object']).columns
        print(f"Encoding these categorical columns: {categorical_columns.tolist()}")
        data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
        return data

    def splitTrainTestData(self, data: pd.DataFrame):
        X = data.drop(columns=['CustomerID', '회사명', '이탈 여부', '가입 일자', '최근 서비스 이용 날짜', '구매 일자'], errors='ignore')
        y = data['이탈 여부']

        # Train-Test Split
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scaleFeatures(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def trainModel(self, X_train, y_train):
        model = LogisticRegression(class_weight='balanced', random_state=42)
        model.fit(X_train, y_train)
        return model

    def evaluateModel(self, model, X_test, y_test):
        y_prediction = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_prediction)
        precision = precision_score(y_test, y_prediction, zero_division=0)
        recall = recall_score(y_test, y_prediction, zero_division=0)
        f1 = f1_score(y_test, y_prediction, zero_division=0)
        confusion = confusion_matrix(y_test, y_prediction)

        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": confusion.tolist(),
        }

        return metrics, y_prediction

    def compareResult(self, y_test, y_prediction):
        return pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})

    def crossValidateModel(self, model, X, y, cv=5):
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        return scores

    def plotFeatureImportance(self, model, feature_names):
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
            sorted_idx = np.argsort(importance)[::-1]
            sorted_features = np.array(feature_names)[sorted_idx]
            sorted_importance = importance[sorted_idx]

            plt.figure(figsize=(10, 6))
            plt.bar(range(len(sorted_importance)), sorted_importance, tick_label=sorted_features)
            plt.xticks(rotation=90)
            plt.title('Feature Importance')
            plt.xlabel('Features')
            plt.ylabel('Coefficient Magnitude')
            plt.show()
