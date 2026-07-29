from fastapi import APIRouter

from app.schemas.health import HealthResponse

api_v1_router = APIRouter()


@api_v1_router.get("/health", response_model=HealthResponse)
def health_v1() -> HealthResponse:
    return HealthResponse(status="ok")
