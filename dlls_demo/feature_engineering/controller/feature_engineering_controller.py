import os
from fastapi import APIRouter, Depends
from dotenv import load_dotenv
from feature_engineering.service.feature_engineering_service_impl import FeatureEngineeringServiceImpl

load_dotenv()
featureEngineeringRouter = APIRouter()

async def injectFeatureEngineeringService() -> FeatureEngineeringServiceImpl:
    return FeatureEngineeringServiceImpl()

@featureEngineeringRouter.post("/feature-engineering")
async def requestFeatureEngineering(
    featureEngineeringService: FeatureEngineeringServiceImpl = Depends(injectFeatureEngineeringService)
):
    # Use environment variable for file path
    file_path = os.getenv("RAW_DATA_PATH", "resource/customer_data.csv")
    print(f"Feature engineering initiated with file: {file_path}")

    # Execute feature engineering process
    featureEngineeringResponse = await featureEngineeringService.feature_engineering(file_path)

    # Extract and format response
    metrics = featureEngineeringResponse["metrics"]
    comparison = featureEngineeringResponse["comparison"].to_dict(orient="records")
    print("Feature engineering completed successfully.")

    return {
        "metrics": metrics,
        "comparison": comparison
    }
