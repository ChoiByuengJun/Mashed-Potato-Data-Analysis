from fastapi import FastAPI
from dotenv import load_dotenv
import uvicorn
import os

from config.cors_config import CorsConfig
from feature_engineering.controller.feature_engineering_controller import featureEngineeringRouter


load_dotenv()

app = FastAPI()

CorsConfig.middlewareConfig(app)

# APIRouter로 작성한 Router를 실제 main에 맵핑
# 결론적으로 다른 도메인에 구성한 라우터를 연결하여 사용할 수 있음
app.include_router(featureEngineeringRouter)

# HOST는 모두에 열려 있고
# FASTAPI_PORT를 통해서 이 서비스가 구동되는 포트 번호를 지정
if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv('HOST'), port=int(os.getenv('FASTAPI_PORT')))
