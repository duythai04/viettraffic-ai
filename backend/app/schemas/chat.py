from typing import Any, Optional

from pydantic import BaseModel, Field


# ============================================================
# REQUEST
# ============================================================

class ChatRequest(BaseModel):
    """
    Dữ liệu frontend gửi lên API.
    """

    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Câu hỏi pháp luật của người dùng",
        examples=[
            "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?"
        ],
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Số lượng evidence tối đa cần truy xuất",
    )


# ============================================================
# SOURCE
# ============================================================

class LegalSource(BaseModel):

    id: Optional[str] = None

    filename: Optional[str] = None

    page: Optional[int] = None

    source: Optional[str] = None

    retrieval_type: Optional[str] = None

    article_number: Optional[Any] = None

    clause_number: Optional[Any] = None

    point_label: Optional[str] = None


# ============================================================
# VALIDATION
# ============================================================

class ValidationInfo(BaseModel):

    status: Optional[str] = None

    confidence: Optional[str] = None

    can_answer: Optional[bool] = None

    stats: Optional[dict[str, Any]] = None


# ============================================================
# USAGE
# ============================================================

class TokenUsage(BaseModel):

    prompt_tokens: Optional[int] = None

    completion_tokens: Optional[int] = None

    total_tokens: Optional[int] = None


# ============================================================
# RESPONSE
# ============================================================

class ChatResponse(BaseModel):

    question: str

    answer: str

    sources: list[LegalSource] = []

    retrieved_count: int = 0

    validation: Optional[ValidationInfo] = None

    penalty_complete: bool = False

    needs_clarification: bool = False

    clarification_question: Optional[str] = None

    model: Optional[str] = None

    usage: Optional[TokenUsage] = None