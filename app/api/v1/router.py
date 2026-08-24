from fastapi import APIRouter, HTTPException
from openai import APIError, OpenAIError

from app.schemas.chat import ChatRequest, ChatResponse
from app.schemas.health import HealthResponse
from app.services import chat as chat_service

api_v1_router = APIRouter()


@api_v1_router.get("/health", response_model=HealthResponse)
def health_v1() -> HealthResponse:
    return HealthResponse(status="ok")


@api_v1_router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        return await chat_service.answer_question(request.question)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (APIError, OpenAIError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "LLM call failed. Check LLM_BASE_URL / LLM_API_KEY / LLM_MODEL "
                f"and that the server is running. ({exc})"
            ),
        ) from exc
