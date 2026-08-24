from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.retrieval.runtime import reset_runtime

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    yield
    reset_runtime()


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        debug=settings.DEBUG,
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(api_v1_router, prefix=settings.API_V1_PREFIX)

    if FRONTEND_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

        @app.get("/")
        def spa() -> FileResponse:
            return FileResponse(FRONTEND_DIR / "index.html")

    return app


app = create_app()
