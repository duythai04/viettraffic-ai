import re
import unicodedata
from typing import Any

from .legal_retriever import (
    analyze_question,
    normalize_text,
    topic_matches_text,
)


# ============================================================
# CONFIG
# ============================================================

VALID_RETRIEVAL_TYPES = {
    "legal_point",
    "legal_clause",
    "legal_article",
    "semantic",
}

PRECISE_RETRIEVAL_TYPES = {
    "legal_point",
    "legal_clause",
}

STRUCTURED_RETRIEVAL_TYPES = {
    "legal_point",
    "legal_clause",
    "legal_article",
}

MIN_PRECISE_EVIDENCE = 1

MIN_ARTICLE_CONTENT_LENGTH = 30
MIN_CLAUSE_CONTENT_LENGTH = 20
MIN_POINT_CONTENT_LENGTH = 10
MIN_SEMANTIC_CONTENT_LENGTH = 40


# ============================================================
# BASIC HELPERS
# ============================================================

def safe_metadata(
    document: dict
) -> dict:
    """
    Lấy metadata an toàn.
    """

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
    """
    Lấy retrieval_type.

    Document semantic cũ có thể không có metadata.
    """

    metadata = safe_metadata(
        document
    )

    retrieval_type = metadata.get(
        "retrieval_type"
    )

    if retrieval_type:
        return retrieval_type

    return "semantic"


def get_document_content(
    document: dict
) -> str:
    """
    Lấy content an toàn.
    """

    content = document.get(
        "content",
        ""
    )

    if content is None:
        return ""

    return str(
        content
    ).strip()


def get_document_filename(
    document: dict
) -> str:
    """
    Lấy filename an toàn.
    """

    filename = document.get(
        "filename",
        ""
    )

    if filename is None:
        return ""

    return str(
        filename
    ).strip()


# ============================================================
# VEHICLE DETECTION IN EVIDENCE
# ============================================================

def detect_vehicle_in_text(
    text: str
) -> set[str]:
    """
    Detect các loại phương tiện xuất hiện trong evidence.

    Trả về set vì một đoạn có thể nhắc nhiều phương tiện.
    """

    content = normalize_text(
        text
    )

    vehicles = set()

    if "xe may chuyen dung" in content:
        vehicles.add(
            "special_vehicle"
        )

    motorcycle_patterns = [
        "xe mo to",
        "xe gan may",
        "xe may",
    ]

    if any(
        pattern in content
        for pattern in motorcycle_patterns
    ):
        vehicles.add(
            "motorcycle"
        )

    car_patterns = [
        "xe o to",
        "xe oto",
        "xe 6 to",
        "oto",
    ]

    if any(
        pattern in content
        for pattern in car_patterns
    ):
        vehicles.add(
            "car"
        )

    bicycle_patterns = [
        "xe dap",
        "xe dap may",
        "xe tho so",
    ]

    if any(
        pattern in content
        for pattern in bicycle_patterns
    ):
        vehicles.add(
            "bicycle"
        )

    return vehicles


def vehicle_matches_evidence(
    question_vehicle: str | None,
    document: dict
) -> bool:
    """
    Kiểm tra evidence có mâu thuẫn rõ ràng với vehicle
    trong câu hỏi hay không.

    Quy tắc:
    - question không có vehicle -> pass.
    - evidence không nhắc vehicle -> chưa đủ để kết luận sai -> pass.
    - evidence có đúng vehicle -> pass.
    - evidence chỉ chứa vehicle khác -> fail.
    """

    if question_vehicle is None:
        return True

    content = get_document_content(
        document
    )

    evidence_vehicles = (
        detect_vehicle_in_text(
            content
        )
    )

    if not evidence_vehicles:
        return True

    if question_vehicle in evidence_vehicles:
        return True

    return False


# ============================================================
# STRUCTURE VALIDATION
# ============================================================

def validate_structure(
    document: dict
) -> tuple[bool, list[str]]:
    """
    Kiểm tra metadata theo retrieval_type.

    legal_point:
        phải có Điều + Khoản + Điểm

    legal_clause:
        phải có Điều + Khoản

    legal_article:
        phải có Điều

    semantic:
        không bắt buộc Điều/Khoản/Điểm
    """

    errors = []

    metadata = safe_metadata(
        document
    )

    retrieval_type = get_retrieval_type(
        document
    )

    if retrieval_type not in VALID_RETRIEVAL_TYPES:
        errors.append(
            f"retrieval_type không hợp lệ: "
            f"{retrieval_type}"
        )

        return False, errors

    article_number = metadata.get(
        "article_number"
    )

    clause_number = metadata.get(
        "clause_number"
    )

    point_label = metadata.get(
        "point_label"
    )

    if retrieval_type == "legal_point":

        if article_number is None:
            errors.append(
                "legal_point thiếu article_number"
            )

        if clause_number is None:
            errors.append(
                "legal_point thiếu clause_number"
            )

        if not point_label:
            errors.append(
                "legal_point thiếu point_label"
            )

    elif retrieval_type == "legal_clause":

        if article_number is None:
            errors.append(
                "legal_clause thiếu article_number"
            )

        if clause_number is None:
            errors.append(
                "legal_clause thiếu clause_number"
            )

    elif retrieval_type == "legal_article":

        if article_number is None:
            errors.append(
                "legal_article thiếu article_number"
            )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# CONTENT VALIDATION
# ============================================================

def validate_content(
    document: dict
) -> tuple[bool, list[str]]:
    """
    Kiểm tra content có tồn tại và đủ dài tối thiểu.
    """

    errors = []

    content = get_document_content(
        document
    )

    retrieval_type = get_retrieval_type(
        document
    )

    if not content:
        return (
            False,
            ["Evidence không có content"]
        )

    content_length = len(
        content
    )

    if (
        retrieval_type == "legal_point"
        and content_length < MIN_POINT_CONTENT_LENGTH
    ):
        errors.append(
            "Nội dung legal_point quá ngắn"
        )

    elif (
        retrieval_type == "legal_clause"
        and content_length < MIN_CLAUSE_CONTENT_LENGTH
    ):
        errors.append(
            "Nội dung legal_clause quá ngắn"
        )

    elif (
        retrieval_type == "legal_article"
        and content_length < MIN_ARTICLE_CONTENT_LENGTH
    ):
        errors.append(
            "Nội dung legal_article quá ngắn"
        )

    elif (
        retrieval_type == "semantic"
        and content_length < MIN_SEMANTIC_CONTENT_LENGTH
    ):
        errors.append(
            "Nội dung semantic quá ngắn"
        )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# SOURCE VALIDATION
# ============================================================

def validate_source(
    document: dict
) -> tuple[bool, list[str]]:
    """
    Evidence structured phải xác định được nguồn văn bản.

    Chấp nhận:
        filename
        hoặc source
    """

    errors = []

    filename = get_document_filename(
        document
    )

    source = document.get(
        "source"
    )

    retrieval_type = get_retrieval_type(
        document
    )

    if retrieval_type in STRUCTURED_RETRIEVAL_TYPES:

        if not filename and not source:
            errors.append(
                "Structured evidence không có source/filename"
            )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# TOPIC VALIDATION
# ============================================================

def validate_topic(
    document: dict,
    topic: str | None
) -> tuple[bool, list[str]]:
    """
    Kiểm tra evidence có thực sự chứa topic của câu hỏi.

    Với legal_point/legal_clause:
        yêu cầu topic match.

    Với legal_article:
        cho phép fallback.

    Với semantic:
        nếu topic rõ thì cũng phải match.
    """

    errors = []

    if topic is None:
        return True, errors

    retrieval_type = get_retrieval_type(
        document
    )

    content = get_document_content(
        document
    )

    topic_match = topic_matches_text(
        content,
        topic
    )

    if retrieval_type in {
        "legal_point",
        "legal_clause",
    }:

        if not topic_match:
            errors.append(
                f"Evidence không match topic: {topic}"
            )

    elif retrieval_type == "semantic":

        if not topic_match:
            errors.append(
                f"Semantic evidence không match topic: "
                f"{topic}"
            )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# VEHICLE VALIDATION
# ============================================================

def validate_vehicle(
    document: dict,
    vehicle: str | None
) -> tuple[bool, list[str]]:
    """
    Kiểm tra evidence có sai phương tiện không.
    """

    errors = []

    if vehicle is None:
        return True, errors

    if not vehicle_matches_evidence(
        vehicle,
        document
    ):
        errors.append(
            f"Evidence không phù hợp vehicle: "
            f"{vehicle}"
        )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# PAGE VALIDATION
# ============================================================

def validate_page_metadata(
    document: dict
) -> tuple[bool, list[str]]:
    """
    Kiểm tra metadata trang.

    Không bắt buộc tuyệt đối vì một số semantic document
    cũ có thể chỉ có document['page'].
    """

    errors = []

    retrieval_type = get_retrieval_type(
        document
    )

    if retrieval_type not in STRUCTURED_RETRIEVAL_TYPES:
        return True, errors

    metadata = safe_metadata(
        document
    )

    start_page = metadata.get(
        "start_page"
    )

    if start_page is None:
        start_page = document.get(
            "page"
        )

    if start_page is None:
        errors.append(
            "Structured evidence thiếu page"
        )

    return (
        len(errors) == 0,
        errors
    )


# ============================================================
# SINGLE DOCUMENT VALIDATION
# ============================================================

def validate_document(
    document: dict,
    analysis: dict
) -> dict:
    """
    Validate một evidence.

    Không mutate document gốc.
    """

    errors = []

    structure_valid, structure_errors = (
        validate_structure(
            document
        )
    )

    errors.extend(
        structure_errors
    )

    content_valid, content_errors = (
        validate_content(
            document
        )
    )

    errors.extend(
        content_errors
    )

    source_valid, source_errors = (
        validate_source(
            document
        )
    )

    errors.extend(
        source_errors
    )

    topic_valid, topic_errors = (
        validate_topic(
            document,
            analysis.get(
                "topic"
            )
        )
    )

    errors.extend(
        topic_errors
    )

    vehicle_valid, vehicle_errors = (
        validate_vehicle(
            document,
            analysis.get(
                "vehicle"
            )
        )
    )

    errors.extend(
        vehicle_errors
    )

    page_valid, page_errors = (
        validate_page_metadata(
            document
        )
    )

    errors.extend(
        page_errors
    )

    is_valid = (
        structure_valid
        and content_valid
        and source_valid
        and topic_valid
        and vehicle_valid
        and page_valid
    )

    return {
        "valid": is_valid,

        "errors": errors,

        "retrieval_type": (
            get_retrieval_type(
                document
            )
        ),

        "document": document,
    }


# ============================================================
# EVIDENCE QUALITY
# ============================================================

def evidence_quality_score(
    document: dict
) -> int:
    """
    Chấm chất lượng evidence.

    Đây KHÔNG phải điểm xác suất.

    Chỉ dùng để sắp xếp:
        Point > Clause > Article > Semantic.
    """

    retrieval_type = get_retrieval_type(
        document
    )

    scores = {
        "legal_point": 100,
        "legal_clause": 80,
        "legal_article": 50,
        "semantic": 20,
    }

    score = scores.get(
        retrieval_type,
        0
    )

    metadata = safe_metadata(
        document
    )

    legal_score = metadata.get(
        "legal_score"
    )

    if isinstance(
        legal_score,
        (int, float)
    ):
        # Chỉ cộng bonus nhỏ.
        score += min(
            int(
                legal_score / 100
            ),
            10
        )

    return score


# ============================================================
# DUPLICATE VALIDATION
# ============================================================

def validation_key(
    document: dict
) -> tuple:
    """
    Key dùng để deduplicate evidence sau validation.
    """

    metadata = safe_metadata(
        document
    )

    return (
        get_document_filename(
            document
        ),

        get_retrieval_type(
            document
        ),

        metadata.get(
            "article_number"
        ),

        metadata.get(
            "clause_number"
        ),

        metadata.get(
            "point_label"
        ),

        normalize_text(
            get_document_content(
                document
            )
        ),
    )


def deduplicate_valid_documents(
    documents: list[dict]
) -> list[dict]:
    """
    Loại evidence trùng sau validation.
    """

    result = []
    seen = set()

    for document in documents:

        key = validation_key(
            document
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            document
        )

    return result


# ============================================================
# VALIDATION STATUS
# ============================================================

def determine_validation_status(
    valid_documents: list[dict],
    analysis: dict
) -> tuple[str, str]:
    """
    Quyết định trạng thái cuối.

    PASS:
        Có Point/Clause chính xác.

    PARTIAL:
        Chỉ có Article hoặc Semantic.

    FAIL:
        Không còn evidence hợp lệ.
    """

    if not valid_documents:
        return (
            "FAIL",
            "Không tìm thấy evidence pháp lý hợp lệ."
        )

    retrieval_types = {
        get_retrieval_type(
            document
        )
        for document in valid_documents
    }

    precise_count = sum(
        1
        for document in valid_documents
        if get_retrieval_type(
            document
        )
        in PRECISE_RETRIEVAL_TYPES
    )

    if (
        precise_count
        >= MIN_PRECISE_EVIDENCE
    ):
        return (
            "PASS",
            "Đã tìm thấy căn cứ pháp lý có cấu trúc "
            "ở mức Khoản/Điểm."
        )

    if "legal_article" in retrieval_types:
        return (
            "PARTIAL",
            "Tìm được Điều luật nhưng chưa xác định "
            "được Khoản/Điểm đủ cụ thể."
        )

    if "semantic" in retrieval_types:
        return (
            "PARTIAL",
            "Chỉ tìm được semantic evidence; "
            "chưa xác định được căn cứ pháp lý "
            "có cấu trúc."
        )

    return (
        "FAIL",
        "Evidence chưa đủ để tạo câu trả lời pháp lý."
    )


# ============================================================
# VALIDATION CONFIDENCE
# ============================================================

def calculate_validation_confidence(
    valid_documents: list[dict],
    status: str
) -> str:
    """
    Confidence dạng nhãn.

    Không biểu diễn xác suất thực tế.
    """

    if status == "FAIL":
        return "low"

    retrieval_types = [
        get_retrieval_type(
            document
        )
        for document in valid_documents
    ]

    if "legal_point" in retrieval_types:
        return "high"

    if "legal_clause" in retrieval_types:
        return "high"

    if "legal_article" in retrieval_types:
        return "medium"

    return "low"


# ============================================================
# MAIN VALIDATOR
# ============================================================

def validate_legal_evidence(
    question: str,
    documents: list[dict]
) -> dict:
    """
    Main Legal Validator.

    Input:
        question
        documents từ legal_retriever.py

    Output:
        {
            status,
            confidence,
            can_answer,
            analysis,
            valid_documents,
            rejected_documents,
            message
        }
    """

    analysis = analyze_question(
        question
    )

    valid_documents = []
    rejected_documents = []

    for document in documents:

        result = validate_document(
            document,
            analysis
        )

        if result["valid"]:

            valid_documents.append(
                document
            )

        else:

            rejected_documents.append({
                "document": document,
                "errors": result[
                    "errors"
                ],
            })

    valid_documents = (
        deduplicate_valid_documents(
            valid_documents
        )
    )

    # Sắp xếp evidence tốt nhất lên trước.
    valid_documents.sort(
        key=evidence_quality_score,
        reverse=True
    )

    status, message = (
        determine_validation_status(
            valid_documents,
            analysis
        )
    )

    confidence = (
        calculate_validation_confidence(
            valid_documents,
            status
        )
    )

    # Chỉ PASS mới cho GPT trả lời trực tiếp.
    can_answer = (
        status == "PASS"
    )

    return {
        "status": status,

        "confidence": confidence,

        "can_answer": can_answer,

        "analysis": analysis,

        "valid_documents": (
            valid_documents
        ),

        "rejected_documents": (
            rejected_documents
        ),

        "message": message,

        "stats": {
            "input_documents": (
                len(documents)
            ),

            "valid_documents": (
                len(valid_documents)
            ),

            "rejected_documents": (
                len(
                    rejected_documents
                )
            ),

            "precise_documents": sum(
                1
                for document
                in valid_documents
                if get_retrieval_type(
                    document
                )
                in PRECISE_RETRIEVAL_TYPES
            ),
        },
    }


# ============================================================
# DEBUG PRINTER
# ============================================================

def print_validation_result(
    result: dict
) -> None:
    """
    In kết quả validation để debug.
    """

    print(
        "\n========== LEGAL VALIDATION =========="
    )

    print(
        "Status:",
        result.get(
            "status"
        )
    )

    print(
        "Confidence:",
        result.get(
            "confidence"
        )
    )

    print(
        "Can answer:",
        result.get(
            "can_answer"
        )
    )

    print(
        "Message:",
        result.get(
            "message"
        )
    )

    print(
        "Stats:",
        result.get(
            "stats"
        )
    )

    print(
        "\nVALID DOCUMENTS:"
    )

    for index, document in enumerate(
        result.get(
            "valid_documents",
            []
        ),
        start=1
    ):

        metadata = safe_metadata(
            document
        )

        print(
            f"[{index}]",
            get_retrieval_type(
                document
            ),
            "|",
            document.get(
                "filename"
            ),
            "| Điều",
            metadata.get(
                "article_number"
            ),
            "| Khoản",
            metadata.get(
                "clause_number"
            ),
            "| Điểm",
            metadata.get(
                "point_label"
            ),
            "| Page",
            metadata.get(
                "start_page",
                document.get(
                    "page"
                )
            ),
        )

    rejected_documents = result.get(
        "rejected_documents",
        []
    )

    if rejected_documents:

        print(
            "\nREJECTED DOCUMENTS:"
        )

        for index, item in enumerate(
            rejected_documents,
            start=1
        ):

            document = item.get(
                "document",
                {}
            )

            metadata = safe_metadata(
                document
            )

            print(
                f"[{index}]",
                get_retrieval_type(
                    document
                ),
                "|",
                document.get(
                    "filename"
                ),
                "| Điều",
                metadata.get(
                    "article_number"
                ),
                "| Errors:",
                item.get(
                    "errors"
                ),
            )