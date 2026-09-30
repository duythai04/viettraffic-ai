from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes.chat import router as chat_router


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="VietTraffic AI API",
    description=(
        "API hỏi đáp pháp luật giao thông "
        "sử dụng Legal RAG."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ============================================================
# ROUTERS
# ============================================================

app.include_router(
    chat_router
)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/",
    tags=["System"],
)
def root():

    return {
        "name": "VietTraffic AI",
        "status": "running",
        "version": "1.0.0",
    }


@app.get(
    "/health",
    tags=["System"],
)
def health():

    return {
        "status": "ok"
    }