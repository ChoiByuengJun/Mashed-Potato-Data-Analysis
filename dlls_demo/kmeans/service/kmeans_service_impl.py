import os
import pandas as pd
from matplotlib import pyplot as plt

from kmeans.repository.kmeans_repository_impl import KMeansRepositoryImpl
from kmeans.service.kmeans_service import KMeansService

class KMeansServiceImpl(KMeansService):
    def __init__(self):
        self.repository = KMeansRepositoryImpl()
        self.data_path = os.getenv("PROCESSED_DATA_PATH", "resource/preprocessed_data.csv")

    async def requestProcess(self):
        # Load Data
        data = self.repository.loadData(self.data_path)

        # Preprocessing for different cluster types
        business_columns = ['업종', '회사 규모', '지역']
        transaction_columns = ['구매 횟수', '평균 구매 금액', '가입 기간']
        product_columns = ['평점', '평균 구매 주기']

        # Business Clustering
        business_data = self.repository.preprocessData(data, business_columns)
        scaled_business, _ = self.repository.scaleData(business_data)
        _, business_labels = self.repository.performKMeans(scaled_business, n_clusters=4)
        data = self.repository.addClusterLabels(data, business_labels, "Business")

        # Transaction Clustering
        transaction_data = data[transaction_columns]
        scaled_transaction, _ = self.repository.scaleData(transaction_data)
        _, transaction_labels = self.repository.performKMeans(scaled_transaction, n_clusters=3)
        data = self.repository.addClusterLabels(data, transaction_labels, "Transaction")

        # Product Clustering
        product_data = self.repository.preprocessData(data, product_columns)
        scaled_product, _ = self.repository.scaleData(product_data)
        _, product_labels = self.repository.performKMeans(scaled_product, n_clusters=5)
        data = self.repository.addClusterLabels(data, product_labels, "Product")

        # Save Results
        output_path = os.getenv("CLUSTERED_DATA_PATH", "resource/clustered_data.csv")
        data.to_csv(output_path, index=False)

        # Visualization Example
        self.visualizeClusters(transaction_data, transaction_labels, "Transaction Clustering", "구매 횟수", "평균 구매 금액")

        return {"message": "K-Means clustering completed successfully", "output_path": output_path}

    def visualizeClusters(self, data, labels, title, x_column, y_column):
        plt.figure(figsize=(8, 6))
        plt.scatter(data[x_column], data[y_column], c=labels, cmap='viridis', alpha=0.5)
        plt.colorbar(label='Cluster')
        plt.title(title)
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.show()
