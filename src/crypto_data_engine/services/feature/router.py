from fastapi import APIRouter, HTTPException
from common_schemas import FeatureQuery
from typing import Any

router = APIRouter()

@router.post("/", summary="Generate or retrieve features")
async def generate_features(q: FeatureQuery):
    # TODO: compute or load features
    return {"data": {}, "params": q.dict()}