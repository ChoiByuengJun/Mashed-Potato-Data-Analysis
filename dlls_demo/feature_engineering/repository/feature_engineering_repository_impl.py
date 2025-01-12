import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from feature_engineering.repository.feature_engineering_repository import FeatureEngineeringRepository

class FeatureEngineeringRepositoryImpl(FeatureEngineeringRepository):
    def handleMissingValues(self, data: pd.DataFrame) -> pd.DataFrame:
        for column in data.columns:
            if data[column].dtype == 'object':
                data[column].fillna(data[column].mode()[0], inplace=True)
            else:
                data[column].fillna(data[column].mean(), inplace=True)
        return data

    def encodeCategoricalFeatures(self, data: pd.DataFrame) -> (pd.DataFrame, dict):
        encoders = {}
        for column in data.select_dtypes(include=['object']).columns:
            encoder = LabelEncoder()
            data[column] = encoder.fit_transform(data[column])
            encoders[column] = encoder
        return data, encoders

    def createNewFeatures(self, data: pd.DataFrame) -> pd.DataFrame:
        data['가입 일자'] = pd.to_datetime(data['가입 일자']) #로우데이터 date타입으로 변경함
        data['최근 서비스 이용 날짜'] = pd.to_datetime(data['최근 서비스 이용 날짜'])
        data['가입 기간'] = (data['최근 서비스 이용 날짜'] - data['가입 일자']).dt.days
        data['서비스 공백 기간'] = (pd.Timestamp.now() - data['최근 서비스 이용 날짜']).dt.days
        data['평균 거래 주기'] = data.groupby('CustomerID')['구매 일자'].diff().dt.days.mean()
        data['평균 거래 금액'] = data['총 거래 금액'] / (data['구매 횟수'] + 1e-9)
        return data

    def splitTrainTestData(self, data: pd.DataFrame):
        X = data.drop(columns=['CustomerID', '이탈 여부', '가입 일자', '최근 서비스 이용 날짜', '구매 일자'])
        y = data['이탈 여부']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def scaleFeatures(self, X_train, X_test):
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    def trainModel(self, X_train, y_train):
        model = LogisticRegression()
        model.fit(X_train, y_train)
        return model

    def evaluateModel(self, model, X_test, y_test):
        y_prediction = model.predict(X_test)
        mseError = mean_squared_error(y_test, y_prediction)
        return mseError, y_prediction

    def compareResult(self, y_test, y_prediction):
        return pd.DataFrame({'Actual': y_test, 'Predicted': y_prediction})
