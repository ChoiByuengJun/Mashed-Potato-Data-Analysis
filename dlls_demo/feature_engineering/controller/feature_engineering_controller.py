import os
from fastapi import APIRouter, Depends

from feature_engineering.service.feature_engineering_service_impl import FeatureEngineeringServiceImpl

featureEngineeringRouter = APIRouter()

async def injectFeatureEngineeringService() -> FeatureEngineeringServiceImpl:
    return FeatureEngineeringServiceImpl()

@featureEngineeringRouter.post("/feature-engineering")
async def requestFeatureEngineering(
    featureEngineeringService: FeatureEngineeringServiceImpl = Depends(injectFeatureEngineeringService)
):
    file_path = os.path.join("resource", "customer_data.csv")
    featureEngineeringResponse = await featureEngineeringService.feature_engineering(file_path)

    comparison = featureEngineeringResponse["comparison"].to_dict(orient="records")
    return {
        "mseError": featureEngineeringResponse["mseError"],
        "comparison": comparison
    }
