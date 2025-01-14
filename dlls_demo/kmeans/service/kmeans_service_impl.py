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
        product_columns = ['평점', '평균 구매 주기','모델명']

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

        # Visualization for all clusters
        self.visualizeClusters(data, business_columns, "Business")
        self.visualizeClusters(data, transaction_columns, "Transaction")
        self.visualizeClusters(data, product_columns, "Product")

        return {"message": "K-Means clustering completed successfully", "output_path": output_path}

    def visualizeClusters(self, data, columns, cluster_type):
        """
        Visualize all combinations of numeric columns for a specific cluster type.
        """
        cluster_label = f"{cluster_type}_Cluster"

        # 숫자형 데이터 필터링
        numeric_data = data[columns].select_dtypes(include=['number'])

        # 숫자형 컬럼이 없을 경우 메시지 출력
        if numeric_data.empty:
            print(f"No numeric data available for {cluster_type} clustering visualization.")
            return

        # 숫자형 데이터 조합으로 산점도 생성
        for i, x_col in enumerate(numeric_data.columns):
            for j, y_col in enumerate(numeric_data.columns):
                if i >= j:  # 중복 및 대각선 방지
                    continue
                plt.figure(figsize=(8, 6))
                plt.scatter(
                    data[x_col],
                    data[y_col],
                    c=data[cluster_label],
                    cmap='viridis',
                    alpha=0.7,
                    edgecolor='k'
                )
                plt.title(f"{cluster_type} Clustering: {x_col} vs {y_col}")
                plt.xlabel(x_col)
                plt.ylabel(y_col)
                plt.colorbar(label='Cluster')
                plt.show()

        # 클러스터별 컬럼 분포 시각화
        for column in numeric_data.columns:
            plt.figure(figsize=(8, 6))
            for cluster in data[cluster_label].unique():
                cluster_data = data[data[cluster_label] == cluster][column]
                cluster_data.plot(kind='kde', label=f"Cluster {cluster}", alpha=0.7)
            plt.title(f"{cluster_type} Clustering: {column} Distribution")
            plt.xlabel(column)
            plt.ylabel("Density")
            plt.legend()
            plt.show()
