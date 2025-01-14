import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pca.repository.pca_repository_impl import PCARepositoryImpl


class PCAServiceImpl:
    def __init__(self):
        self.repository = PCARepositoryImpl()
        self.data_path = os.getenv("PROCESSED_DATA_PATH", "resource/preprocessed_data.csv")

    async def performPCA(self):
        # 데이터 로드
        data = self.repository.loadData(self.data_path)

        # 상관관계 분석 시각화
        self._createCorrelationHeatmap(data)

        # 범주형 데이터 인코딩
        data = self.repository.encodeCategoricalFeatures(data)
        print("Categorical features encoded.")

        # 숫자형 데이터만 선택
        numeric_data = data.select_dtypes(include=["number"])

        # PCA 적용 전에 필요 없는 컬럼 제거
        if 'CustomerID' in numeric_data.columns:
            numeric_data = numeric_data.drop(columns=['CustomerID'])

        # 데이터 스케일링
        scaled_data, _ = self.repository.scaleData(numeric_data)

        # PCA 적용
        n_components = min(5, scaled_data.shape[1])  # 최대 5개 또는 컬럼 수 제한
        pca, transformed_data, explained_variance, components = self.repository.applyPCA(
            scaled_data, numeric_data.columns, n_components
        )

        # 결과 저장
        output_path = os.getenv("PCA_RESULT_PATH", "resource/pca_result.csv")
        transformed_df = pd.DataFrame(transformed_data, columns=[f"PC{i + 1}" for i in range(n_components)])
        transformed_df.to_csv(output_path, index=False)

        # 주성분 가중치 히트맵 생성
        self._createComponentHeatmap(components, numeric_data.columns)

        # PCA 설명 분산 시각화
        self._createExplainedVariancePlot(explained_variance)

        # 결과 반환 수정
        return {
            "message": "PCA completed successfully",
            "explained_variance": explained_variance.tolist(),  # NumPy 배열의 경우 .tolist() 사용
            "components": components.values.tolist(),  # DataFrame을 리스트로 변환
            "output_path": output_path
        }

    def _createCorrelationHeatmap(self, data):
        """상관관계 분석 히트맵 생성"""
        # 숫자형 데이터만 선택
        numeric_data = data.select_dtypes(include=["number"])

        # 상관관계 계산
        correlation_matrix = numeric_data.corr()

        # 히트맵 시각화
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Feature Correlation Heatmap")
        plt.tight_layout()

        # 결과 저장 및 표시
        plt.savefig("resource/correlation_heatmap.png")
        plt.show()
        print("Correlation heatmap saved to resource/correlation_heatmap.png.")

    def _createComponentHeatmap(self, components, feature_names):
        """주성분 가중치 히트맵 생성"""
        plt.figure(figsize=(12, 8))
        heatmap_data = pd.DataFrame(
            components,
            columns=feature_names,
            index=[f"PC{i + 1}" for i in range(components.shape[0])]
        )

        # 상위 중요한 피처 10개만 선택
        top_features = heatmap_data.abs().sum().sort_values(ascending=False).head(10).index
        heatmap_data = heatmap_data[top_features]

        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-0.5,
            vmax=0.5,
            cbar=True
        )
        plt.title("PCA Component Heatmap")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.yticks(fontsize=10)
        plt.tight_layout()
        plt.savefig("resource/component_heatmap.png")
        plt.show()
        print("Component heatmap saved to resource/component_heatmap.png.")

    def _createExplainedVariancePlot(self, explained_variance):
        """PCA 설명 분산 비율 시각화"""
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
        plt.title("Explained Variance by Principal Components")
        plt.xlabel("Principal Component")
        plt.ylabel("Explained Variance Ratio")
        plt.grid()
        plt.tight_layout()
        plt.savefig("resource/explained_variance_plot.png")
        plt.show()
        print("Explained variance plot saved to resource/explained_variance_plot.png.")
