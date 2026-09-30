import logging

from fastapi import APIRouter, HTTPException

from backend.app.schemas.chat import (
    ChatRequest,
    ChatResponse,
)

from backend.app.services.rag.rag_chain import (
    ask_rag,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/api",
    tags=["Chat"],
)


# ============================================================
# CHAT
# ============================================================

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Hỏi đáp pháp luật giao thông",
)
def chat(
    request: ChatRequest,
) -> ChatResponse:

    question = request.message.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Câu hỏi không được để trống.",
        )

    try:

        result = ask_rag(
            question=question,
            top_k=request.top_k,
        )

        return ChatResponse(
            question=result.get(
                "question",
                question,
            ),

            answer=result.get(
                "answer",
                "",
            ),

            sources=result.get(
                "sources",
                [],
            ),

            retrieved_count=result.get(
                "retrieved_count",
                0,
            ),

            validation=result.get(
                "validation",
            ),

            penalty_complete=result.get(
                "penalty_complete",
                False,
            ),

            needs_clarification=result.get(
                "needs_clarification",
                False,
            ),

            clarification_question=result.get(
                "clarification_question",
            ),

            model=result.get(
                "model",
            ),

            usage=result.get(
                "usage",
            ),
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    except Exception as error:

        logger.exception(
            "Lỗi khi xử lý câu hỏi RAG"
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Hệ thống gặp lỗi khi xử lý "
                "câu hỏi pháp luật."
            ),
        ) from error