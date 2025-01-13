from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import os
import matplotlib.pyplot as plt
from matplotlib import rc
from config.cors_config import CorsConfig
from feature_engineering.controller.feature_engineering_controller import featureEngineeringRouter
from kmeans.controller.kmeans_controller import kMeansRouter
from pca.controller.pca_controller import pcaRouter

load_dotenv()


# macOS 한글 폰트 설정
rc('font', family='Apple SD Gothic Neo') # 윈도우는 폰트 깨지면 나눔이였나로 하면됨.

# 마이너스 기호 깨짐 방지
plt.rcParams['axes.unicode_minus'] = False

# 기본 폰트 크기와 그래프 크기 설정 (선택)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (8, 6)


# 전역적으로 사용할 파일 경로 환경 변수 설정

#raw_data_path = os.getenv("RAW_DATA_PATH", "resource/customer_data.csv")
#file_path = os.getenv("PREPROCESSED_DATA_PATH", "resource/customer_data.csv")

app = FastAPI()

CorsConfig.middlewareConfig(app)

# APIRouter로 작성한 Router를 실제 main에 맵핑
# 결론적으로 다른 도메인에 구성한 라우터를 연결하여 사용할 수 있음
app.include_router(featureEngineeringRouter)
app.include_router(kMeansRouter)
app.include_router(pcaRouter)

# HOST는 모두에 열려 있고
# FASTAPI_PORT를 통해서 이 서비스가 구동되는 포트 번호를 지정
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv('HOST'), port=int(os.getenv('FASTAPI_PORT')))
