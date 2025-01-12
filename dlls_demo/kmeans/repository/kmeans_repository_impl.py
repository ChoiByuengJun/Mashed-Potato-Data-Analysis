import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from kmeans.repository.kmeans_repository import KMeansRepository

class KMeansRepositoryImpl(KMeansRepository):
    def loadData(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def preprocessData(self, data: pd.DataFrame, columns: list) -> pd.DataFrame:
        return pd.get_dummies(data[columns], drop_first=True)

    def scaleData(self, data: pd.DataFrame) -> (pd.DataFrame, object):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def performKMeans(self, scaled_data: pd.DataFrame, n_clusters: int) -> (object, list):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        return kmeans, labels

    def addClusterLabels(self, data: pd.DataFrame, labels: list, cluster_type: str) -> pd.DataFrame:
        data[f"{cluster_type}_Cluster"] = labels
        return data
