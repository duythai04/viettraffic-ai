
import json
import re
import unicodedata

from pathlib import Path

from .retriever import retrieve_documents


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

INDEX_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "legal_index.json"
)

DECREE_168 = "168-nd-cp.signed.pdf"
DECREE_238 = "238-ndcp.signed.pdf"

MAX_SEMANTIC_DOCUMENTS = 5
MAX_LEGAL_ARTICLES = 2


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa để tìm kiếm, không thay đổi nội dung pháp luật gốc.
    """

    text = unicodedata.normalize(
        "NFD",
        str(text).lower()
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.replace("đ", "d")

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


# ============================================================
# LOAD LEGAL INDEX
# ============================================================

def load_legal_index() -> dict:

    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {INDEX_PATH}. "
            "Hãy chạy legal_parser.py trước."
        )

    with open(
        INDEX_PATH,
        "r",
        encoding="utf-8"
    ) as file:
        data = json.load(file)

    if "articles" not in data:
        raise ValueError(
            "legal_index.json không có trường 'articles'."
        )

    return data


# ============================================================
# QUESTION ANALYSIS
# ============================================================

def classify_vehicle(question: str) -> str | None:

    text = normalize_text(question)

    # Kiểm tra xe máy chuyên dùng trước xe máy thông thường.
    if "xe may chuyen dung" in text:
        return "special_vehicle"

    motorcycle_patterns = [
        "xe may",
        "xe mo to",
        "xe gan may",
        "mo to",
    ]

    if any(
        pattern in text
        for pattern in motorcycle_patterns
    ):
        return "motorcycle"

    car_patterns = [
        "xe o to",
        "o to",
        "oto",
        "xe hoi",
    ]

    if any(
        pattern in text
        for pattern in car_patterns
    ):
        return "car"

    return None


def classify_article(article: dict) -> str | None:
    """
    Phân loại theo TIÊU ĐỀ điều luật.

    Không dùng toàn bộ content để phân loại, vì nội dung một điều
    có thể nhắc đến nhiều loại phương tiện khác nhau.
    """

    title = normalize_text(
        article.get("article_title", "")
    )

    if "xe may chuyen dung" in title:
        return "special_vehicle"

    # Lỗi OCR đã quan sát được: 'xe 6 tô'.
    car_patterns = [
        "xe o to",
        "xe 6 to",
        "xe oto",
    ]

    if any(
        pattern in title
        for pattern in car_patterns
    ):
        return "car"

    motorcycle_patterns = [
        "xe mo to",
        "xe gan may",
    ]

    if any(
        pattern in title
        for pattern in motorcycle_patterns
    ):
        return "motorcycle"

    return None


def detect_topic(question: str) -> str | None:

    text = normalize_text(question)

    traffic_light_patterns = [
        "vuot den do",
        "den do",
        "den tin hieu",
        "tin hieu giao thong",
    ]

    if any(
        pattern in text
        for pattern in traffic_light_patterns
    ):
        return "traffic_light"

    child_patterns = [
        "tre em",
        "ghe tre em",
        "thiet bi an toan",
    ]

    if any(
        pattern in text
        for pattern in child_patterns
    ):
        return "child_safety"

    return None


def is_decree_238_overview(question: str) -> bool:

    text = normalize_text(question)

    if "238" not in text:
        return False

    overview_patterns = [
        "sua doi",
        "bo sung",
        "noi dung nao",
        "nhung noi dung",
        "tong quan",
        "tom tat",
    ]

    return any(
        pattern in text
        for pattern in overview_patterns
    )


# ============================================================
# VEHICLE FILTER
# ============================================================

def article_matches_vehicle(
    article: dict,
    vehicle_type: str | None
) -> bool:

    if vehicle_type is None:
        return True

    article_vehicle = classify_article(article)

    # Điều luật chung vẫn có thể là nguồn bổ sung.
    if article_vehicle is None:
        return True

    return article_vehicle == vehicle_type


# ============================================================
# TOPIC MATCHING
# ============================================================

def has_traffic_light_violation(text: str) -> bool:
    """
    Chỉ nhận diện hành vi không chấp hành tín hiệu đèn.

    Không coi việc 'che khuất đèn tín hiệu' là vượt đèn đỏ.
    """

    content = normalize_text(text)

    # Trường hợp OCR tương đối rõ.
    exact_patterns = [
        "khong chap hanh hieu lenh cua den tin hieu giao thong",
        "khong chap hanh hieu lenh den tin hieu giao thong",
        "khong chap hanh tin hieu den",
    ]

    if any(
        pattern in content
        for pattern in exact_patterns
    ):
        return True

    # Cho phép một số từ nằm giữa các thành phần câu.
    # Dùng giới hạn ký tự để tránh khớp các đoạn không liên quan.
    flexible_pattern = (
        r"khong\s+chap\s+hanh"
        r".{0,65}"
        r"(?:den\s+tin\s+hieu|tin\s+hieu\s+den)"
    )

    if re.search(
        flexible_pattern,
        content
    ):
        return True

    return False


def has_child_safety_rule(text: str) -> bool:

    content = normalize_text(text)

    patterns = [
        "tre em duoi 10 tuoi",
        "thiet bi an toan phu hop",
        "cho tre em",
        "dua don tre em",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def score_clause(
    clause: dict,
    topic: str | None
) -> int:

    if topic is None:
        return 0

    content = clause.get(
        "content",
        ""
    )

    if topic == "traffic_light":

        if has_traffic_light_violation(content):
            return 100

        return 0

    if topic == "child_safety":

        if has_child_safety_rule(content):
            return 100

        return 0

    return 0


def find_relevant_clauses(
    article: dict,
    topic: str | None
) -> list[dict]:

    matches = []

    for clause in article.get(
        "clauses",
        []
    ):

        score = score_clause(
            clause,
            topic
        )

        if score <= 0:
            continue

        matches.append({
            **clause,
            "match_score": score,
        })

    matches.sort(
        key=lambda item: (
            -item["match_score"],
            item.get("clause_number", 999),
        )
    )

    return matches


# ============================================================
# ARTICLE SEARCH
# ============================================================

def search_legal_articles(
    question: str,
    top_k: int = 3
) -> list[dict]:

    index_data = load_legal_index()

    articles = index_data.get(
        "articles",
        []
    )

    vehicle_type = classify_vehicle(
        question
    )

    topic = detect_topic(
        question
    )

    # Câu hỏi tổng quan văn bản sửa đổi cần đọc toàn văn 238.
    # Không chọn ngẫu nhiên hai điều từ legal index.
    if is_decree_238_overview(question):
        return []

    candidates = []

    for article in articles:

        filename = article.get(
            "filename",
            ""
        )

        article_vehicle = classify_article(
            article
        )

        # Với câu hỏi vượt đèn đỏ, tìm trong nghị định xử phạt.
        if (
            topic == "traffic_light"
            and filename != DECREE_168
        ):
            continue

        if not article_matches_vehicle(
            article,
            vehicle_type
        ):
            continue

        relevant_clauses = find_relevant_clauses(
            article,
            topic
        )

        exact_vehicle = (
            vehicle_type is not None
            and article_vehicle == vehicle_type
        )

        # Điều về phương tiện khác đã bị loại ở trên.
        # Điều không xác định phương tiện chỉ được dùng khi
        # có khoản khớp hành vi.
        if (
            topic == "traffic_light"
            and not exact_vehicle
            and not relevant_clauses
        ):
            continue

        score = 0

        if exact_vehicle:
            score += 200

        if relevant_clauses:
            score += 100

        if (
            topic == "child_safety"
            and filename != DECREE_168
        ):
            score += 10

        if score <= 0:
            continue

        candidates.append({
            **article,
            "legal_score": score,
            "relevant_clauses": relevant_clauses,
            "vehicle_match": exact_vehicle,
        })

    # Ưu tiên điều đúng loại phương tiện trước.
    # Sau đó mới ưu tiên khoản khớp hành vi.
    candidates.sort(
        key=lambda item: (
            item["vehicle_match"],
            bool(item["relevant_clauses"]),
            item["legal_score"],
        ),
        reverse=True
    )

    # Câu hỏi về phương tiện cụ thể:
    # Nếu đã tìm thấy điều đúng phương tiện, không để
    # điều chung chen vào top_k.
    if (
        topic == "traffic_light"
        and vehicle_type is not None
    ):

        exact_candidates = [
            item
            for item in candidates
            if item["vehicle_match"]
        ]

        if exact_candidates:
            candidates = exact_candidates

    return candidates[:top_k]


# ============================================================
# DOCUMENT CONVERSION
# ============================================================

def clause_to_document(
    article: dict,
    clause: dict
) -> dict:

    content = (
        f"{article['article_title']}\n"
        f"Khoản {clause['clause_number']}\n"
        f"{clause['content']}"
    )

    return {
        "content": content,
        "source": article["source"],
        "filename": article["filename"],
        "page": clause["start_page"],
        "score": None,
        "metadata": {
            "retrieval_type": "legal_clause",
            "article_number": article["article_number"],
            "clause_number": clause["clause_number"],
            "start_page": clause["start_page"],
            "end_page": clause["end_page"],
            "original_source": article.get(
                "original_source"
            ),
        },
    }


def article_to_document(
    article: dict
) -> dict:

    return {
        "content": article["content"],
        "source": article["source"],
        "filename": article["filename"],
        "page": article["start_page"],
        "score": None,
        "metadata": {
            "retrieval_type": "legal_article",
            "article_number": article["article_number"],
            "start_page": article["start_page"],
            "end_page": article["end_page"],
            "original_source": article.get(
                "original_source"
            ),
        },
    }


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_documents(
    documents: list[dict]
) -> list[dict]:

    unique_documents = []

    seen = set()

    for document in documents:

        metadata = document.get(
            "metadata",
            {}
        ) or {}

        key = (
            document.get("filename"),
            document.get("page"),
            metadata.get("article_number"),
            metadata.get("clause_number"),
            document.get("content"),
        )

        if key in seen:
            continue

        seen.add(key)

        unique_documents.append(
            document
        )

    return unique_documents


# ============================================================
# COMBINED RETRIEVAL
# ============================================================

def retrieve_legal_documents(
    question: str,
    top_k: int = 5,
    article_top_k: int = 2
) -> list[dict]:

    print(
        "\n========== LEGAL RETRIEVAL =========="
    )

    overview_238 = is_decree_238_overview(
        question
    )

    # --------------------------------------------------------
    # MODE 1: FULL DECREE 238
    # --------------------------------------------------------

    if overview_238:

        print(
            "Retrieval mode: FULL DECREE 238"
        )

        # Retriever hiện tại đã có nhánh lấy toàn bộ 238.
        semantic_documents = retrieve_documents(
            question=question,
            top_k=top_k
        )

        result = deduplicate_documents(
            semantic_documents
        )

        print(
            "Tổng tài liệu sau khi gộp:",
            len(result)
        )

        return result

    # --------------------------------------------------------
    # MODE 2: LEGAL + SEMANTIC
    # --------------------------------------------------------

    print(
        "Retrieval mode: HYBRID LEGAL SEARCH"
    )

    vehicle_type = classify_vehicle(
        question
    )

    topic = detect_topic(
        question
    )

    articles = search_legal_articles(
        question=question,
        top_k=article_top_k
    )

    print(
        "Số điều luật tìm thấy:",
        len(articles)
    )

    legal_documents = []

    for article in articles:

        relevant_clauses = article.get(
            "relevant_clauses",
            []
        )

        # Nếu parser tìm được đúng khoản, dùng khoản đó.
        if relevant_clauses:

            for clause in relevant_clauses[:2]:

                legal_documents.append(
                    clause_to_document(
                        article,
                        clause
                    )
                )

        else:

            # OCR có thể làm hỏng câu chứa hành vi.
            # Với điều đã khớp đúng loại phương tiện,
            # giữ toàn bộ điều thay vì đoán khoản khác.
            legal_documents.append(
                article_to_document(
                    article
                )
            )

    # --------------------------------------------------------
    # SEMANTIC FALLBACK
    # --------------------------------------------------------

    semantic_documents = retrieve_documents(
        question=question,
        top_k=top_k
    )

    # Với câu hỏi vượt đèn đỏ và đã tìm được điều đúng
    # phương tiện, không đưa các chunks từ điều khác
    # lên trước nguồn pháp luật đã chọn.
    if (
        topic == "traffic_light"
        and vehicle_type is not None
        and legal_documents
    ):

        semantic_documents = [
            document
            for document in semantic_documents
            if (
                document.get("filename") == DECREE_238
            )
        ][:2]

    elif legal_documents:

        semantic_documents = (
            semantic_documents[
                :MAX_SEMANTIC_DOCUMENTS
            ]
        )

    combined = (
        legal_documents
        + semantic_documents
    )

    result = deduplicate_documents(
        combined
    )

    print(
        "Tổng tài liệu sau khi gộp:",
        len(result)
    )

    return result
