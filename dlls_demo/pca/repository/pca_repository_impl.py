import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from pca.repository.pca_repository import PCARepository

class PCARepositoryImpl(PCARepository):
    def loadData(self, file_path: str) -> pd.DataFrame:
        return pd.read_csv(file_path)

    def scaleData(self, data: pd.DataFrame):
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        return scaled_data, scaler

    def applyPCA(self, scaled_data, data_columns, n_components: int):
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(scaled_data)
        explained_variance = pca.explained_variance_ratio_
        components = pd.DataFrame(
            pca.components_,
            columns=data_columns,
            index=[f"PC{i+1}" for i in range(n_components)]
        )
        return pca, transformed_data, explained_variance, components

    def createHeatmap(self, components, features):
        plt.figure(figsize=(10, 6))
        sns.heatmap(components, annot=True, cmap='coolwarm', xticklabels=features, yticklabels=components.index)
        plt.title("Principal Components Heatmap")
        plt.xlabel("Features")
        plt.ylabel("Principal Components")
        plt.show()
