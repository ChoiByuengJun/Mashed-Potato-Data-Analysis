import pandas as pd
from matplotlib import pyplot as plt

from kmeans.repository.kmeans_repository_impl import KMeansRepositoryImpl
from kmeans.service.kmeans_service import KMeansService


class KMeansServiceImpl(KMeansService):

    def __init__(self):
        self.__kMeansRepository = KMeansRepositoryImpl()

    async def requestProcess(self):
        createdData = self.__kMeansRepository.createData()
        addOnCreatedData = self.__kMeansRepository.appendAgeGroup20Data(createdData)
        addOnCreatedData = self.__kMeansRepository.appendAgeGroup30Data(addOnCreatedData)
        addOnCreatedData = self.__kMeansRepository.appendAgeGroup40Data(addOnCreatedData)
        addOnCreatedData = self.__kMeansRepository.appendAgeGroup50Data(addOnCreatedData)

        dataFrame = pd.DataFrame(addOnCreatedData)

        X = self.__kMeansRepository.prepareData(dataFrame)
        scaler, scaledX = self.__kMeansRepository.scaleData(X)

        kmeans, dataFrame = self.__kMeansRepository.trainingKMeans(scaledX, dataFrame)

        print(f"클러스터 중심: {kmeans.cluster_centers_}")
        print(f"클러스터 별 데이터 개수: {dataFrame['Cluster'].value_counts()}")

        plt.figure(figsize=(8, 6))
        plt.scatter(X['FPS'], X['RPG'], c=dataFrame['Cluster'], cmap='viridis', alpha=0.5)
        centroids = kmeans.cluster_centers_
        centroids_unscaled = scaler.inverse_transform(centroids)

        plt.scatter(
            centroids_unscaled[:, 0], centroids_unscaled[:, 1],
            c='red', marker='x', s=200, label='Centroids'
        )
        plt.title("FPS와 RPG기반 K-means 클러스터링")
        plt.xlabel("FPS")
        plt.xlabel("RPG")
        plt.legend()
        plt.show()

