import os
import re

from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .legal_retriever import (
    retrieve_legal_documents,
    analyze_question,
    normalize_text,
)

from .legal_validator import (
    validate_legal_evidence,
)


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

ENV_PATH = (
    BASE_DIR
    / "backend"
    / ".env"
)

load_dotenv(
    ENV_PATH
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)

MAX_CONTEXT_CHARS = 55000

MAX_SINGLE_CHUNK_CHARS = 30000


if not OPENAI_API_KEY:
    raise RuntimeError(
        "Không tìm thấy OPENAI_API_KEY "
        "trong backend/.env"
    )


client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=60.0,
    max_retries=2
)


# ============================================================
# 2. SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
Bạn là VietTraffic AI, trợ lý hỗ trợ tra cứu pháp luật
giao thông đường bộ Việt Nam.

Bạn CHỈ được trả lời dựa trên TÀI LIỆU THAM KHẢO
được cung cấp trong prompt.

==================================================
QUY TẮC BẮT BUỘC
==================================================

1. Không sử dụng kiến thức bên ngoài tài liệu tham khảo
để tự bổ sung:

- mức tiền phạt;
- số Điều;
- số Khoản;
- số Điểm;
- số điểm giấy phép lái xe bị trừ;
- thời hạn;
- ngày có hiệu lực;
- hình thức xử phạt;
- biện pháp khắc phục hậu quả.

2. Không được suy đoán thông tin pháp luật.

Nếu tài liệu không đủ căn cứ thì phải nói rõ
chưa đủ căn cứ.

3. Phải phân biệt chính xác đối tượng áp dụng:

- xe ô tô;
- xe mô tô;
- xe gắn máy;
- xe máy chuyên dùng;
- xe đạp;
- xe đạp máy;
- các loại phương tiện khác.

4. Tuyệt đối không lấy quy định áp dụng cho một loại
phương tiện để trả lời cho loại phương tiện khác.

Ví dụ:

Không lấy mức phạt của xe máy chuyên dùng để trả lời
cho xe mô tô hoặc xe gắn máy thông thường.

==================================================
QUY TẮC VỀ MỨC PHẠT
==================================================

5. Khi người dùng hỏi mức xử phạt, phải xác định được:

- loại phương tiện;
- hành vi vi phạm;
- Điều;
- Khoản;
- Điểm nếu có;
- nội dung của Khoản quy định mức phạt.

6. Một Điểm luật chỉ xác định hành vi chưa chắc đã đủ
để xác định mức tiền phạt.

Nếu mức tiền nằm ở phần mở đầu của Khoản thì phải dựa
trên phần nội dung Khoản cha được cung cấp.

7. Chỉ được kết luận mức tiền phạt khi:

- hành vi nằm trong Điểm/Khoản được cung cấp;
- mức tiền được thể hiện trong cùng Khoản áp dụng;
- có thể xác định rõ chúng thuộc cùng Điều và Khoản.

8. Nếu tài liệu chỉ cho biết hành vi nhưng không cung cấp
được mức tiền tương ứng thì không được tự bổ sung mức tiền.

==================================================
VĂN BẢN SỬA ĐỔI
==================================================

9. Nếu tài liệu tham khảo chứa văn bản sửa đổi, bổ sung:

- phải xem xét nội dung sửa đổi được cung cấp;
- không mặc định quy định cũ vẫn còn nguyên;
- không tự kết luận tình trạng hiệu lực nếu tài liệu
  chưa cung cấp đủ thông tin.

10. Nếu có mâu thuẫn giữa các nguồn được cung cấp,
không tự quyết định nguồn nào hiện hành nếu context
không đủ dữ liệu để xác định.

Hãy nêu rõ sự khác nhau.

==================================================
OCR
==================================================

11. Một số tài liệu được OCR nên có thể sai:

- dấu tiếng Việt;
- ký tự;
- số;
- tên văn bản;
- cách xuống dòng.

Không được tự sửa số tiền hoặc số Điều/Khoản/Điểm
bằng suy đoán.

Có thể diễn đạt lại lỗi dấu rõ ràng nếu việc đó
không làm thay đổi ý nghĩa pháp lý.

==================================================
TRÍCH DẪN
==================================================

12. Tài liệu được đánh dấu:

[S1]
[S2]
[S3]
...

Chỉ được sử dụng các mã nguồn thực sự xuất hiện
trong TÀI LIỆU THAM KHẢO.

13. Mỗi nhận định pháp lý quan trọng phải có nguồn.

Ví dụ:

Theo điểm c khoản 7 Điều 7 ... [S1]

14. Không được tạo nguồn không tồn tại.

15. Khi có đủ metadata, ưu tiên nêu:

- tên văn bản;
- Điều;
- Khoản;
- Điểm;
- trang PDF.

==================================================
CÁCH TRẢ LỜI
==================================================

16. Trả lời trực tiếp câu hỏi trước.

17. Sau đó nêu căn cứ pháp lý.

18. Không cần lặp lại toàn bộ văn bản pháp luật nếu
không cần thiết.

19. Nếu câu hỏi thiếu thông tin để xác định chính xác
mức xử phạt, hãy nói rõ thông tin người dùng cần bổ sung.

Ví dụ:

"Bạn cần cho biết xe chạy vượt quá tốc độ cho phép
bao nhiêu km/h để xác định chính xác mức xử phạt."

20. Không đưa ra kết luận vượt quá nội dung của
TÀI LIỆU THAM KHẢO.
"""


# ============================================================
# 3. DOCUMENT HELPERS
# ============================================================

def get_metadata(
    document: dict
) -> dict:

    metadata = document.get(
        "metadata",
        {}
    )

    if not isinstance(
        metadata,
        dict
    ):
        return {}

    return metadata


def get_retrieval_type(
    document: dict
) -> str:

    metadata = get_metadata(
        document
    )

    return metadata.get(
        "retrieval_type",
        "semantic"
    )


def get_page(
    document: dict
):

    metadata = get_metadata(
        document
    )

    return metadata.get(
        "start_page",
        document.get(
            "page"
        )
    )


# ============================================================
# 4. PENALTY HELPERS
# ============================================================

def contains_money_penalty(
    text: str
) -> bool:
    """
    Kiểm tra text có dấu hiệu chứa khung tiền phạt không.

    Không cố parse số tiền ở đây.
    Chỉ kiểm tra evidence có phần mức tiền hay chưa.
    """

    if not text:
        return False

    normalized = normalize_text(
        text
    )

    patterns = [
        r"phat\s+tien",
        r"tu\s+[\d\.,]+\s*(?:dong|trieu)",
        r"den\s+[\d\.,]+\s*(?:dong|trieu)",
    ]

    return any(
        re.search(
            pattern,
            normalized
        )
        is not None
        for pattern in patterns
    )


def get_parent_clause_content(
    document: dict
) -> str:
    """
    Lấy toàn bộ nội dung Khoản cha.

    legal_point:
        lấy metadata.parent_clause_content

    legal_clause:
        content chính là Khoản.

    Các loại khác:
        trả về chuỗi rỗng.
    """

    metadata = get_metadata(
        document
    )

    retrieval_type = (
        get_retrieval_type(
            document
        )
    )

    if retrieval_type == "legal_point":

        return (
            metadata.get(
                "parent_clause_content"
            )
            or ""
        ).strip()

    if retrieval_type == "legal_clause":

        return (
            document.get(
                "content"
            )
            or ""
        ).strip()

    return ""


def check_penalty_completeness(
    analysis: dict,
    documents: list[dict]
) -> dict:
    """
    Kiểm tra evidence có đủ để trả lời câu hỏi mức phạt hay không.

    Đây KHÔNG phải validator pháp lý đầy đủ.

    Nó chỉ kiểm tra vấn đề quan trọng:

        hành vi
            +
        Khoản cha
            +
        dấu hiệu mức tiền

    có cùng structured evidence hay không.
    """

    intent = analysis.get(
        "intent"
    )

    if intent != "penalty":

        return {
            "required": False,
            "complete": True,
            "reason": None,
            "supporting_documents": [],
        }

    supporting_documents = []

    for document in documents:

        retrieval_type = (
            get_retrieval_type(
                document
            )
        )

        if retrieval_type not in {
            "legal_point",
            "legal_clause",
        }:
            continue

        parent_clause = (
            get_parent_clause_content(
                document
            )
        )

        if not parent_clause:
            continue

        if contains_money_penalty(
            parent_clause
        ):
            supporting_documents.append(
                document
            )

    if supporting_documents:

        return {
            "required": True,
            "complete": True,
            "reason": (
                "Đã tìm thấy structured evidence "
                "có nội dung Khoản chứa mức tiền."
            ),
            "supporting_documents": (
                supporting_documents
            ),
        }

    return {
        "required": True,
        "complete": False,
        "reason": (
            "Đã xác định được hành vi nhưng chưa tìm thấy "
            "đầy đủ nội dung Khoản chứa mức tiền xử phạt."
        ),
        "supporting_documents": [],
    }


# ============================================================
# 5. CLARIFICATION
# ============================================================

def detect_clarification_need(
    question: str,
    analysis: dict,
    documents: list[dict]
) -> dict:
    """
    Phát hiện một số câu hỏi chưa đủ dữ kiện.

    Hiện tại ưu tiên case tốc độ.

    Ví dụ:
        "Xe máy chạy quá tốc độ bị phạt thế nào?"

    Không thể chọn duy nhất một mức phạt nếu chưa biết
    vượt bao nhiêu km/h.
    """

    topic = analysis.get(
        "topic"
    )

    intent = analysis.get(
        "intent"
    )

    normalized_question = normalize_text(
        question
    )

    if (
        topic == "speed"
        and intent == "penalty"
    ):

        speed_value_pattern = (
            r"\b\d+(?:[.,]\d+)?\s*"
            r"(?:km/h|kmh|km)\b"
        )

        has_speed_value = (
            re.search(
                speed_value_pattern,
                normalized_question
            )
            is not None
        )

        if not has_speed_value:

            return {
                "needed": True,

                "question": (
                    "Bạn chạy vượt quá tốc độ cho phép "
                    "bao nhiêu km/h?"
                ),

                "reason": (
                    "Mức xử phạt quá tốc độ phụ thuộc "
                    "vào mức vượt quá tốc độ cho phép."
                ),
            }

    return {
        "needed": False,
        "question": None,
        "reason": None,
    }


# ============================================================
# 6. CONTEXT BUILDER
# ============================================================

def build_context(
    documents: list[dict],
    max_chars: int = MAX_CONTEXT_CHARS
):
    """
    Build context cho GPT.

    legal_point đặc biệt:

        Point
        +
        Parent Clause

    để GPT có cả:
        hành vi
        +
        mức phạt trong cùng Khoản.
    """

    context_parts = []
    sources = []

    used_chars = 0

    for document in documents:

        content = (
            document.get(
                "content"
            )
            or ""
        ).strip()

        if not content:
            continue

        metadata = get_metadata(
            document
        )

        retrieval_type = (
            get_retrieval_type(
                document
            )
        )

        filename = document.get(
            "filename",
            "Không rõ"
        )

        page = get_page(
            document
        )

        article_number = metadata.get(
            "article_number"
        )

        clause_number = metadata.get(
            "clause_number"
        )

        point_label = metadata.get(
            "point_label"
        )

        parent_clause_content = (
            metadata.get(
                "parent_clause_content"
            )
            or ""
        ).strip()

        source_id = (
            f"S{len(sources) + 1}"
        )

        # ----------------------------------------------------
        # STRUCTURED LEGAL POINT
        # ----------------------------------------------------

        if retrieval_type == "legal_point":

            legal_structure = []

            if article_number is not None:
                legal_structure.append(
                    f"Điều {article_number}"
                )

            if clause_number is not None:
                legal_structure.append(
                    f"Khoản {clause_number}"
                )

            if point_label:
                legal_structure.append(
                    f"Điểm {point_label}"
                )

            structure_text = (
                " - ".join(
                    legal_structure
                )
                if legal_structure
                else "Không rõ"
            )

            block = (
                f"[{source_id}]\n"
                f"Loại nguồn: LEGAL_POINT\n"
                f"Tên văn bản: {filename}\n"
                f"Cấu trúc: {structure_text}\n"
                f"Trang PDF: "
                f"{page if page is not None else 'Không rõ'}\n"
            )

            if parent_clause_content:

                block += (
                    "\n"
                    "TOÀN BỘ KHOẢN CHA:\n"
                    f"{parent_clause_content}\n"
                )

            block += (
                "\n"
                "ĐIỂM ĐƯỢC TRUY XUẤT:\n"
                f"{content}\n"
                f"[/{source_id}]"
            )

        # ----------------------------------------------------
        # OTHER DOCUMENT TYPES
        # ----------------------------------------------------

        else:

            structure_parts = []

            if article_number is not None:
                structure_parts.append(
                    f"Điều {article_number}"
                )

            if clause_number is not None:
                structure_parts.append(
                    f"Khoản {clause_number}"
                )

            if point_label:
                structure_parts.append(
                    f"Điểm {point_label}"
                )

            structure_text = (
                " - ".join(
                    structure_parts
                )
                if structure_parts
                else "Không rõ"
            )

            block = (
                f"[{source_id}]\n"
                f"Loại nguồn: {retrieval_type}\n"
                f"Tên văn bản: {filename}\n"
                f"Cấu trúc: {structure_text}\n"
                f"Trang PDF: "
                f"{page if page is not None else 'Không rõ'}\n"
                f"Nội dung:\n"
                f"{content}\n"
                f"[/{source_id}]"
            )

        if len(
            block
        ) > MAX_SINGLE_CHUNK_CHARS:

            continue

        if (
            used_chars
            + len(block)
            > max_chars
        ):
            break

        context_parts.append(
            block
        )

        sources.append({
            "id": source_id,

            "filename": filename,

            "page": page,

            "source": document.get(
                "source"
            ),

            "distance": document.get(
                "score"
            ),

            "retrieval_type": (
                retrieval_type
            ),

            "article_number": (
                article_number
            ),

            "clause_number": (
                clause_number
            ),

            "point_label": (
                point_label
            ),

            "content": content,
        })

        used_chars += len(
            block
        )

    return (
        "\n\n".join(
            context_parts
        ),
        sources
    )


# ============================================================
# 7. CITATIONS
# ============================================================

def extract_cited_sources(
    answer: str,
    sources: list[dict]
) -> list[dict]:

    cited_ids = set(
        re.findall(
            r"\[(S\d+)\]",
            answer
        )
    )

    return [
        {
            "id": source[
                "id"
            ],

            "filename": source[
                "filename"
            ],

            "page": source[
                "page"
            ],

            "source": source[
                "source"
            ],

            "retrieval_type": source[
                "retrieval_type"
            ],

            "article_number": source[
                "article_number"
            ],

            "clause_number": source[
                "clause_number"
            ],

            "point_label": source[
                "point_label"
            ],
        }

        for source in sources

        if source[
            "id"
        ] in cited_ids
    ]


# ============================================================
# 8. SPECIAL INSTRUCTIONS
# ============================================================

def build_special_instructions(
    analysis: dict,
    penalty_check: dict
) -> str:

    instructions = []

    vehicle = analysis.get(
        "vehicle"
    )

    topic = analysis.get(
        "topic"
    )

    intent = analysis.get(
        "intent"
    )

    if vehicle == "motorcycle":

        instructions.append(
            "Người dùng hỏi về xe mô tô/xe gắn máy "
            "thông thường. Không áp dụng quy định của "
            "xe máy chuyên dùng."
        )

    elif vehicle == "car":

        instructions.append(
            "Người dùng hỏi về xe ô tô. Chỉ sử dụng "
            "quy định phù hợp với xe ô tô."
        )

    elif vehicle == "special_vehicle":

        instructions.append(
            "Người dùng hỏi về xe máy chuyên dùng. "
            "Không áp dụng mức phạt của xe mô tô thông thường."
        )

    if intent == "penalty":

        instructions.append(
            "Đây là câu hỏi về xử phạt. Chỉ nêu mức tiền "
            "nếu mức tiền xuất hiện trong nội dung Khoản "
            "được cung cấp."
        )

    if topic == "speed":

        instructions.append(
            "Mức xử phạt quá tốc độ có thể phụ thuộc vào "
            "số km/h vượt quá tốc độ cho phép. Không tự chọn "
            "một khung nếu câu hỏi chưa đủ dữ kiện."
        )

    if penalty_check.get(
        "complete"
    ):

        instructions.append(
            "Penalty completeness check đã xác nhận có "
            "structured evidence chứa nội dung Khoản có "
            "dấu hiệu mức tiền. Tuy nhiên vẫn phải đọc "
            "chính xác tài liệu trước khi trả lời."
        )

    if not instructions:

        instructions.append(
            "Áp dụng các quy tắc trong system prompt."
        )

    return "\n".join(
        f"- {instruction}"
        for instruction in instructions
    )


# ============================================================
# 9. SAFE RESPONSE HELPERS
# ============================================================

def build_insufficient_response(
    question: str,
    reason: str,
    retrieved_count: int = 0,
    validation: dict | None = None
) -> dict:

    answer = (
        "Tôi chưa tìm thấy đủ căn cứ trong tài liệu "
        "được cung cấp để trả lời chính xác câu hỏi này."
    )

    if reason:
        answer += (
            "\n\n"
            f"Lý do: {reason}"
        )

    return {
        "question": question,

        "answer": answer,

        "sources": [],

        "retrieved_sources": [],

        "retrieved_count": (
            retrieved_count
        ),

        "validation": validation,

        "penalty_complete": False,

        "needs_clarification": False,

        "clarification_question": None,

        "model": None,

        "usage": None,
    }


def build_clarification_response(
    question: str,
    clarification: dict,
    documents: list[dict],
    validation: dict
) -> dict:

    clarification_question = (
        clarification.get(
            "question"
        )
    )

    reason = clarification.get(
        "reason"
    )

    answer = (
        f"{reason}\n\n"
        f"{clarification_question}"
    )

    return {
        "question": question,

        "answer": answer,

        "sources": [],

        "retrieved_sources": [],

        "retrieved_count": len(
            documents
        ),

        "validation": {
            "status": validation.get(
                "status"
            ),

            "confidence": validation.get(
                "confidence"
            ),

            "can_answer": validation.get(
                "can_answer"
            ),

            "stats": validation.get(
                "stats"
            ),
        },

        "penalty_complete": False,

        "needs_clarification": True,

        "clarification_question": (
            clarification_question
        ),

        "model": None,

        "usage": None,
    }


# ============================================================
# 10. MAIN RAG PIPELINE
# ============================================================

def ask_rag(
    question: str,
    top_k: int = 5
) -> dict:

    # --------------------------------------------------------
    # VALIDATE INPUT
    # --------------------------------------------------------

    if not isinstance(
        question,
        str
    ):
        raise ValueError(
            "question phải là chuỗi."
        )

    question = question.strip()

    if not question:
        raise ValueError(
            "Câu hỏi không được để trống."
        )

    if top_k < 1:
        raise ValueError(
            "top_k phải >= 1."
        )

    print(
        "\n========== QUESTION ANALYSIS =========="
    )

    analysis = analyze_question(
        question
    )

    print(
        "Question:",
        question
    )

    print(
        "Vehicle:",
        analysis.get(
            "vehicle"
        )
    )

    print(
        "Topic:",
        analysis.get(
            "topic"
        )
    )

    print(
        "Intent:",
        analysis.get(
            "intent"
        )
    )

    # --------------------------------------------------------
    # 1. RETRIEVER
    # --------------------------------------------------------

    print(
        "\n========== RETRIEVING =========="
    )

    documents = retrieve_legal_documents(
        question=question,
        top_k=top_k,
        article_top_k=3
    )

    print(
        f"Đã truy xuất {len(documents)} documents."
    )

    if not documents:

        return build_insufficient_response(
            question=question,

            reason=(
                "Retriever không tìm thấy "
                "căn cứ pháp lý phù hợp."
            ),

            retrieved_count=0
        )

    # --------------------------------------------------------
    # 2. LEGAL VALIDATOR
    # --------------------------------------------------------

    print(
        "\n========== VALIDATING =========="
    )

    validation = (
        validate_legal_evidence(
            question=question,
            documents=documents
        )
    )

    print(
        "Status:",
        validation.get(
            "status"
        )
    )

    print(
        "Confidence:",
        validation.get(
            "confidence"
        )
    )

    print(
        "Can answer:",
        validation.get(
            "can_answer"
        )
    )

    valid_documents = (
        validation.get(
            "valid_documents",
            []
        )
    )

    if not validation.get(
        "can_answer"
    ):

        return build_insufficient_response(
            question=question,

            reason=validation.get(
                "message",
                "Legal Validator không xác nhận evidence."
            ),

            retrieved_count=len(
                documents
            ),

            validation={
                "status": validation.get(
                    "status"
                ),

                "confidence": validation.get(
                    "confidence"
                ),

                "can_answer": validation.get(
                    "can_answer"
                ),

                "stats": validation.get(
                    "stats"
                ),
            }
        )

    if not valid_documents:

        return build_insufficient_response(
            question=question,

            reason=(
                "Validator không còn evidence hợp lệ."
            ),

            retrieved_count=len(
                documents
            )
        )

    # --------------------------------------------------------
    # 3. CLARIFICATION CHECK
    # --------------------------------------------------------

    clarification = (
        detect_clarification_need(
            question=question,
            analysis=analysis,
            documents=valid_documents
        )
    )

    if clarification.get(
        "needed"
    ):

        print(
            "\n========== CLARIFICATION REQUIRED =========="
        )

        print(
            clarification.get(
                "question"
            )
        )

        return build_clarification_response(
            question=question,
            clarification=clarification,
            documents=valid_documents,
            validation=validation
        )

    # --------------------------------------------------------
    # 4. PENALTY COMPLETENESS
    # --------------------------------------------------------

    print(
        "\n========== PENALTY CHECK =========="
    )

    penalty_check = (
        check_penalty_completeness(
            analysis=analysis,
            documents=valid_documents
        )
    )

    print(
        "Required:",
        penalty_check.get(
            "required"
        )
    )

    print(
        "Complete:",
        penalty_check.get(
            "complete"
        )
    )

    print(
        "Reason:",
        penalty_check.get(
            "reason"
        )
    )

    if (
        penalty_check.get(
            "required"
        )
        and not penalty_check.get(
            "complete"
        )
    ):

        return build_insufficient_response(
            question=question,

            reason=penalty_check.get(
                "reason"
            ),

            retrieved_count=len(
                documents
            ),

            validation={
                "status": validation.get(
                    "status"
                ),

                "confidence": validation.get(
                    "confidence"
                ),

                "can_answer": validation.get(
                    "can_answer"
                ),

                "stats": validation.get(
                    "stats"
                ),
            }
        )

    # --------------------------------------------------------
    # 5. BUILD CONTEXT
    # --------------------------------------------------------

    print(
        "\n========== BUILDING CONTEXT =========="
    )

    context, sources = build_context(
        valid_documents
    )

    if not context:

        return build_insufficient_response(
            question=question,

            reason=(
                "Evidence hợp lệ nhưng không tạo được "
                "context cho mô hình."
            ),

            retrieved_count=len(
                documents
            )
        )

    print(
        "Context sources:",
        len(
            sources
        )
    )

    print(
        "Context chars:",
        len(
            context
        )
    )

    # --------------------------------------------------------
    # 6. SPECIAL INSTRUCTIONS
    # --------------------------------------------------------

    extra_rules = (
        build_special_instructions(
            analysis=analysis,
            penalty_check=penalty_check
        )
    )

    # --------------------------------------------------------
    # 7. USER PROMPT
    # --------------------------------------------------------

    user_prompt = f"""
TÀI LIỆU THAM KHẢO:

{context}

==================================================

PHÂN TÍCH CÂU HỎI:

Loại phương tiện:
{analysis.get("vehicle")}

Chủ đề:
{analysis.get("topic")}

Ý định:
{analysis.get("intent")}

==================================================

CÂU HỎI:

{question}

==================================================

YÊU CẦU BỔ SUNG:

{extra_rules}

==================================================

Hãy trả lời câu hỏi chỉ dựa trên tài liệu được cung cấp.

Mỗi nhận định pháp lý quan trọng phải dẫn nguồn bằng
mã [S1], [S2]... tương ứng với tài liệu tham khảo.

Không sử dụng kiến thức bên ngoài context để bổ sung
mức phạt hoặc căn cứ pháp lý.
"""

    # --------------------------------------------------------
    # 8. GPT
    # --------------------------------------------------------

    print(
        "\n========== GENERATING =========="
    )

    response = (
        client.chat.completions.create(
            model=MODEL_NAME,

            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },

                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            temperature=0.1,

            max_tokens=1800,
        )
    )

    answer = (
        response
        .choices[0]
        .message
        .content
        or ""
    ).strip()

    # --------------------------------------------------------
    # 9. CITATION EXTRACTION
    # --------------------------------------------------------

    cited_sources = (
        extract_cited_sources(
            answer=answer,
            sources=sources
        )
    )

    # --------------------------------------------------------
    # 10. FINAL RESULT
    # --------------------------------------------------------

    return {
        "question": question,

        "answer": answer,

        "sources": (
            cited_sources
        ),

        "retrieved_sources": [
            {
                "id": source[
                    "id"
                ],

                "filename": source[
                    "filename"
                ],

                "page": source[
                    "page"
                ],

                "source": source[
                    "source"
                ],

                "distance": source[
                    "distance"
                ],

                "retrieval_type": source[
                    "retrieval_type"
                ],

                "article_number": source[
                    "article_number"
                ],

                "clause_number": source[
                    "clause_number"
                ],

                "point_label": source[
                    "point_label"
                ],
            }

            for source in sources
        ],

        "retrieved_count": len(
            valid_documents
        ),

        "validation": {
            "status": validation.get(
                "status"
            ),

            "confidence": validation.get(
                "confidence"
            ),

            "can_answer": validation.get(
                "can_answer"
            ),

            "stats": validation.get(
                "stats"
            ),
        },

        "penalty_complete": (
            penalty_check.get(
                "complete"
            )
        ),

        "needs_clarification": False,

        "clarification_question": None,

        "model": response.model,

        "usage": (
            {
                "prompt_tokens": (
                    response.usage.prompt_tokens
                ),

                "completion_tokens": (
                    response.usage.completion_tokens
                ),

                "total_tokens": (
                    response.usage.total_tokens
                ),
            }

            if response.usage

            else None
        ),
    }


# ============================================================
# 11. LOCAL TEST
# ============================================================

if __name__ == "__main__":

    questions = [
        (
            "Vượt đèn đỏ bằng xe máy "
            "bị xử phạt thế nào?"
        ),

        (
            "Vượt đèn đỏ bằng ô tô "
            "bị xử phạt thế nào?"
        ),

        (
            "Chở trẻ em dưới 10 tuổi trên ô tô "
            "cần tuân thủ quy định gì?"
        ),

        (
            "Xe máy chạy quá tốc độ "
            "bị phạt thế nào?"
        ),
    ]

    for question in questions:

        print(
            "\n"
            + "=" * 100
        )

        print(
            "CÂU HỎI:",
            question
        )

        print(
            "=" * 100
        )

        try:

            result = ask_rag(
                question=question,
                top_k=5
            )

            print(
                "\nCÂU TRẢ LỜI:"
            )

            print(
                result[
                    "answer"
                ]
            )

            print(
                "\nVALIDATION:"
            )

            print(
                result.get(
                    "validation"
                )
            )

            print(
                "\nPENALTY COMPLETE:"
            )

            print(
                result.get(
                    "penalty_complete"
                )
            )

            print(
                "\nNEEDS CLARIFICATION:"
            )

            print(
                result.get(
                    "needs_clarification"
                )
            )

            if result.get(
                "clarification_question"
            ):

                print(
                    "\nCLARIFICATION:"
                )

                print(
                    result[
                        "clarification_question"
                    ]
                )

            print(
                "\nNGUỒN ĐƯỢC GPT TRÍCH DẪN:"
            )

            for source in result.get(
                "sources",
                []
            ):

                print(
                    f"[{source['id']}] "
                    f"{source['filename']} "
                    f"| Điều "
                    f"{source['article_number']} "
                    f"| Khoản "
                    f"{source['clause_number']} "
                    f"| Điểm "
                    f"{source['point_label']} "
                    f"| Trang "
                    f"{source['page']}"
                )

            print(
                "\nSỐ EVIDENCE:",
                result.get(
                    "retrieved_count"
                )
            )

            if result.get(
                "usage"
            ):

                print(
                    "TỔNG TOKENS:",
                    result[
                        "usage"
                    ][
                        "total_tokens"
                    ]
                )

        except Exception as error:

            print(
                "\nERROR:"
            )

            print(
                type(
                    error
                ).__name__,
                ":",
                error
            )