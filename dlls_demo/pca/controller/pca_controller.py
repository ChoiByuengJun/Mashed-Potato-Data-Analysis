import os
from fastapi import APIRouter, Depends
from pca.service.pca_service_impl import PCAServiceImpl

pcaRouter = APIRouter()

async def injectPCAService() -> PCAServiceImpl:
    return PCAServiceImpl()

@pcaRouter.post("/pca")
async def requestPCA(pcaService: PCAServiceImpl = Depends(injectPCAService)):
    response = await pcaService.performPCA()
    return response
