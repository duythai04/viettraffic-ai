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

MAX_SEMANTIC_DOCUMENTS = 3
MAX_LEGAL_ARTICLES = 3
MAX_CLAUSES_PER_ARTICLE = 5
MAX_POINTS_PER_CLAUSE = 5


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa tiếng Việt để phục vụ matching.

    Ví dụ:
        "Xe mô tô" -> "xe mo to"
        "Đèn tín hiệu" -> "den tin hieu"
    """

    if not text:
        return ""

    text = unicodedata.normalize(
        "NFD",
        str(text).lower()
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.replace(
        "đ",
        "d"
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD LEGAL INDEX
# ============================================================

def load_legal_index() -> dict:
    """
    Load legal_index.json do legal_parser.py tạo.
    """

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
# QUESTION ANALYZER
# ============================================================

def classify_vehicle(
    question: str
) -> str | None:
    """
    Phân loại phương tiện được nhắc tới trong câu hỏi.
    """

    text = normalize_text(
        question
    )

    # Phải kiểm tra trước "xe máy chuyên dùng"
    # vì nó chứa "xe máy".
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

    bicycle_patterns = [
        "xe dap",
        "xe dap may",
    ]

    if any(
        pattern in text
        for pattern in bicycle_patterns
    ):
        return "bicycle"

    return None


def detect_topic(
    question: str
) -> str | None:
    """
    Xác định chủ đề pháp luật chính của câu hỏi.
    """

    text = normalize_text(
        question
    )

    topics = {
        "traffic_light": [
            "vuot den do",
            "den do",
            "den tin hieu",
            "tin hieu giao thong",
            "khong chap hanh den",
        ],

        "child_safety": [
            "tre em",
            "ghe tre em",
            "thiet bi an toan cho tre",
            "thiet bi an toan phu hop",
            "duoi 10 tuoi",
            "1,35 met",
            "1.35 met",
        ],

        "speed": [
            "qua toc do",
            "vuot qua toc do",
            "chay qua toc do",
            "toc do toi da",
            "toc do quy dinh",
        ],

        "alcohol": [
            "nong do con",
            "ruou bia",
            "uong ruou",
            "uong bia",
            "hoi tho",
        ],

        "helmet": [
            "mu bao hiem",
            "khong doi mu",
            "khong doi mu bao hiem",
        ],

        "wrong_lane": [
            "sai lan",
            "di sai lan",
            "lan duong",
            "khong dung lan",
            "khong dung phan duong",
        ],

        "license": [
            "giay phep lai xe",
            "bang lai",
            "khong co bang lai",
            "khong co giay phep lai xe",
        ],
    }

    for topic, patterns in topics.items():
        if any(
            pattern in text
            for pattern in patterns
        ):
            return topic

    return None


def detect_intent(
    question: str
) -> str:
    """
    Xác định mục đích câu hỏi.

    penalty:
        hỏi mức phạt

    rule:
        hỏi quy định

    overview:
        hỏi tổng quan / sửa đổi / bổ sung

    general:
        các câu hỏi còn lại
    """

    text = normalize_text(
        question
    )

    penalty_patterns = [
        "phat bao nhieu",
        "bi phat",
        "xu phat",
        "muc phat",
        "phat tien",
        "tru diem",
        "bi xu ly",
    ]

    if any(
        pattern in text
        for pattern in penalty_patterns
    ):
        return "penalty"

    rule_patterns = [
        "quy dinh",
        "duoc phep",
        "co duoc",
        "can tuan thu",
        "phai lam gi",
        "can lam gi",
    ]

    if any(
        pattern in text
        for pattern in rule_patterns
    ):
        return "rule"

    overview_patterns = [
        "tom tat",
        "tong quan",
        "noi dung nao",
        "nhung noi dung",
        "sua doi",
        "bo sung",
    ]

    if any(
        pattern in text
        for pattern in overview_patterns
    ):
        return "overview"

    return "general"


def analyze_question(
    question: str
) -> dict:
    """
    Phân tích câu hỏi thành các thành phần phục vụ retrieval.
    """

    return {
        "question": question,

        "normalized_question": normalize_text(
            question
        ),

        "vehicle": classify_vehicle(
            question
        ),

        "topic": detect_topic(
            question
        ),

        "intent": detect_intent(
            question
        ),
    }


# ============================================================
# DECREE 238 OVERVIEW
# ============================================================

def is_decree_238_overview(
    question: str
) -> bool:
    """
    Detect câu hỏi kiểu:

        Nghị định 238 sửa đổi những nội dung nào?
        Tóm tắt Nghị định 238.
    """

    analysis = analyze_question(
        question
    )

    text = analysis[
        "normalized_question"
    ]

    return (
        "238" in text
        and analysis["intent"] == "overview"
    )


# ============================================================
# ARTICLE VEHICLE CLASSIFICATION
# ============================================================

def classify_article_vehicle(
    article: dict
) -> str | None:
    """
    Xác định Điều luật dành cho loại phương tiện nào.

    Ưu tiên article_title.

    Có thêm một phần content ngắn để chịu lỗi OCR khi title
    bị xuống dòng.
    """

    title = normalize_text(
        article.get(
            "article_title",
            ""
        )
    )

    content_preview = normalize_text(
        article.get(
            "content",
            ""
        )[:500]
    )

    searchable = (
        title
        + " "
        + content_preview
    )

    if "xe may chuyen dung" in title:
        return "special_vehicle"

    motorcycle_patterns = [
        "xe mo to",
        "xe gan may",
    ]

    if any(
        pattern in title
        for pattern in motorcycle_patterns
    ):
        return "motorcycle"

    # OCR có thể đọc "xe ô tô" thành "xe 6 tô".
    car_patterns = [
        "xe o to",
        "xe oto",
        "xe 6 to",
    ]

    if any(
        pattern in title
        for pattern in car_patterns
    ):
        return "car"

    bicycle_patterns = [
        "xe dap",
        "xe tho so",
    ]

    if any(
        pattern in title
        for pattern in bicycle_patterns
    ):
        return "bicycle"

    # --------------------------------------------------------
    # OCR FALLBACK
    # --------------------------------------------------------
    # Chỉ dùng content preview khi title không đủ thông tin.
    # --------------------------------------------------------

    if "xe may chuyen dung" in searchable:
        return "special_vehicle"

    if any(
        pattern in searchable
        for pattern in motorcycle_patterns
    ):
        return "motorcycle"

    if any(
        pattern in searchable
        for pattern in car_patterns
    ):
        return "car"

    if any(
        pattern in searchable
        for pattern in bicycle_patterns
    ):
        return "bicycle"

    return None


def article_matches_vehicle(
    article: dict,
    vehicle: str | None
) -> bool:
    """
    Kiểm tra Điều luật có phù hợp loại phương tiện không.
    """

    if vehicle is None:
        return True

    article_vehicle = (
        classify_article_vehicle(
            article
        )
    )

    # Điều chung vẫn có thể liên quan.
    if article_vehicle is None:
        return True

    return article_vehicle == vehicle


# ============================================================
# DOCUMENT FILTER
# ============================================================

def document_allowed(
    filename: str,
    analysis: dict
) -> bool:
    """
    Giới hạn văn bản theo intent.

    Với câu hỏi xử phạt, ưu tiên các Nghị định xử phạt.
    """

    topic = analysis.get(
        "topic"
    )

    intent = analysis.get(
        "intent"
    )

    if (
        intent == "penalty"
        and topic is not None
    ):
        return filename in {
            DECREE_168,
            DECREE_238,
        }

    return True


# ============================================================
# TOPIC MATCHERS
# ============================================================

def has_traffic_light_violation(
    text: str
) -> bool:
    """
    Nhận diện hành vi liên quan đèn tín hiệu giao thông.
    """

    content = normalize_text(
        text
    )

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

    flexible_patterns = [
        (
            r"khong\s+chap\s+hanh"
            r".{0,80}"
            r"(?:den\s+tin\s+hieu|tin\s+hieu\s+den)"
        ),

        (
            r"(?:den\s+do|den\s+tin\s+hieu)"
            r".{0,80}"
            r"khong\s+chap\s+hanh"
        ),
    ]

    return any(
        re.search(
            pattern,
            content
        )
        is not None
        for pattern in flexible_patterns
    )


def has_child_safety_rule(
    text: str
) -> bool:
    """
    Nhận diện quy định cụ thể về trẻ em dưới 10 tuổi /
    chiều cao / thiết bị an toàn.

    Không dùng pattern quá rộng như "chở trẻ em" vì dễ
    match nhầm xe đưa đón học sinh.
    """

    content = normalize_text(
        text
    )

    strong_patterns = [
        "tre em duoi 10 tuoi",
        "duoi 1,35 met",
        "duoi 1.35 met",
        "thiet bi an toan phu hop cho tre em",
        "thiet bi an toan phu hop",
    ]

    return any(
        pattern in content
        for pattern in strong_patterns
    )


def has_speed_violation(
    text: str
) -> bool:
    """
    Nhận diện hành vi quá tốc độ.
    """

    content = normalize_text(
        text
    )

    patterns = [
        "vuot qua toc do quy dinh",
        "chay qua toc do quy dinh",
        "toc do quy dinh",
        "qua toc do",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def has_alcohol_violation(
    text: str
) -> bool:
    """
    Nhận diện hành vi liên quan nồng độ cồn.
    """

    content = normalize_text(
        text
    )

    patterns = [
        "nong do con",
        "trong mau hoac hoi tho",
        "trong mau hoac trong hoi tho",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def has_helmet_violation(
    text: str
) -> bool:
    """
    Nhận diện hành vi liên quan mũ bảo hiểm.
    """

    content = normalize_text(
        text
    )

    patterns = [
        "mu bao hiem",
        "khong doi mu bao hiem",
        "doi mu bao hiem",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def has_wrong_lane_violation(
    text: str
) -> bool:
    """
    Nhận diện hành vi sai làn / phần đường.
    """

    content = normalize_text(
        text
    )

    patterns = [
        "khong dung lan duong",
        "di sai lan duong",
        "khong dung phan duong",
        "lan duong",
        "phan duong",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def has_license_rule(
    text: str
) -> bool:
    """
    Nhận diện nội dung liên quan GPLX.
    """

    content = normalize_text(
        text
    )

    patterns = [
        "khong co giay phep lai xe",
        "giay phep lai xe khong phu hop",
        "giay phep lai xe",
    ]

    return any(
        pattern in content
        for pattern in patterns
    )


def topic_matches_text(
    text: str,
    topic: str | None
) -> bool:
    """
    Kiểm tra text có match topic không.
    """

    if topic is None:
        return False

    matchers = {
        "traffic_light": (
            has_traffic_light_violation
        ),

        "child_safety": (
            has_child_safety_rule
        ),

        "speed": (
            has_speed_violation
        ),

        "alcohol": (
            has_alcohol_violation
        ),

        "helmet": (
            has_helmet_violation
        ),

        "wrong_lane": (
            has_wrong_lane_violation
        ),

        "license": (
            has_license_rule
        ),
    }

    matcher = matchers.get(
        topic
    )

    if matcher is None:
        return False

    return matcher(
        text
    )


# ============================================================
# POINT RETRIEVAL
# ============================================================

def score_point(
    point: dict,
    analysis: dict
) -> int:
    """
    Chấm điểm một Điểm luật.
    """

    topic = analysis.get(
        "topic"
    )

    content = point.get(
        "content",
        ""
    )

    if topic_matches_text(
        content,
        topic
    ):
        return 300

    return 0


def find_relevant_points(
    clause: dict,
    analysis: dict
) -> list[dict]:
    """
    Tìm các Điểm phù hợp trong một Khoản.
    """

    matches = []

    for point in clause.get(
        "points",
        []
    ):
        score = score_point(
            point,
            analysis
        )

        if score <= 0:
            continue

        matches.append({
            **point,
            "match_score": score,
        })

    matches.sort(
        key=lambda item: (
            -item["match_score"],
            str(
                item.get(
                    "point_label",
                    ""
                )
            ),
        )
    )

    return matches


# ============================================================
# CLAUSE RETRIEVAL
# ============================================================

def score_clause(
    clause: dict,
    analysis: dict
) -> int:
    """
    Chấm điểm Khoản.

    +200 nếu toàn Khoản match topic.
    +100 nếu bên trong có Point match.
    """

    topic = analysis.get(
        "topic"
    )

    score = 0

    content = clause.get(
        "content",
        ""
    )

    if topic_matches_text(
        content,
        topic
    ):
        score += 200

    relevant_points = (
        find_relevant_points(
            clause,
            analysis
        )
    )

    if relevant_points:
        score += 100

    return score


def find_relevant_clauses(
    article: dict,
    analysis: dict
) -> list[dict]:
    """
    Tìm các Khoản phù hợp trong một Điều.
    """

    matches = []

    for clause in article.get(
        "clauses",
        []
    ):
        score = score_clause(
            clause,
            analysis
        )

        if score <= 0:
            continue

        relevant_points = (
            find_relevant_points(
                clause,
                analysis
            )
        )

        matches.append({
            **clause,

            "match_score": score,

            "relevant_points": (
                relevant_points
            ),
        })

    matches.sort(
        key=lambda item: (
            -item["match_score"],
            item.get(
                "clause_number",
                999
            ),
        )
    )

    return matches


# ============================================================
# ARTICLE SCORING
# ============================================================

def score_article(
    article: dict,
    analysis: dict
) -> dict | None:
    """
    Chấm điểm Điều luật.

    Ưu tiên:
        đúng vehicle
        +
        có relevant clause/point
    """

    filename = article.get(
        "filename",
        ""
    )

    if not document_allowed(
        filename,
        analysis
    ):
        return None

    vehicle = analysis.get(
        "vehicle"
    )

    article_vehicle = (
        classify_article_vehicle(
            article
        )
    )

    if not article_matches_vehicle(
        article,
        vehicle
    ):
        return None

    relevant_clauses = (
        find_relevant_clauses(
            article,
            analysis
        )
    )

    score = 0

    exact_vehicle = (
        vehicle is not None
        and article_vehicle == vehicle
    )

    if exact_vehicle:
        score += 400

    if relevant_clauses:
        score += 300

        score += min(
            relevant_clauses[0][
                "match_score"
            ],
            300
        )

    # Topic rõ nhưng Điều chung không chứa evidence
    # => không có lý do giữ Điều đó.
    if (
        analysis.get("topic") is not None
        and not exact_vehicle
        and not relevant_clauses
    ):
        return None

    # Điều đúng vehicle nhưng OCR có thể làm hỏng nội dung.
    # Chỉ giữ với score thấp để làm fallback.
    if (
        exact_vehicle
        and not relevant_clauses
    ):
        score += 50

    if score <= 0:
        return None

    return {
        **article,

        "legal_score": score,

        "vehicle_match": (
            exact_vehicle
        ),

        "article_vehicle": (
            article_vehicle
        ),

        "relevant_clauses": (
            relevant_clauses
        ),
    }


# ============================================================
# ARTICLE SEARCH
# ============================================================

def search_legal_articles(
    question: str,
    top_k: int = MAX_LEGAL_ARTICLES
) -> list[dict]:
    """
    Tìm Điều luật phù hợp nhất.

    Nếu đã có Điều chứa structured evidence thì loại những
    Article fallback chỉ match vehicle nhưng không match topic.
    """

    if is_decree_238_overview(
        question
    ):
        return []

    index_data = load_legal_index()

    analysis = analyze_question(
        question
    )

    candidates = []

    for article in index_data.get(
        "articles",
        []
    ):
        result = score_article(
            article,
            analysis
        )

        if result is None:
            continue

        candidates.append(
            result
        )

    candidates.sort(
        key=lambda item: (
            item.get(
                "legal_score",
                0
            ),

            item.get(
                "vehicle_match",
                False
            ),

            bool(
                item.get(
                    "relevant_clauses"
                )
            ),
        ),
        reverse=True
    )

    vehicle = analysis.get(
        "vehicle"
    )

    # ========================================================
    # VEHICLE FILTER
    # ========================================================

    if vehicle is not None:
        exact_vehicle_candidates = [
            candidate
            for candidate in candidates
            if candidate.get(
                "vehicle_match"
            )
        ]

        if exact_vehicle_candidates:
            candidates = (
                exact_vehicle_candidates
            )

    # ========================================================
    # STRUCTURED EVIDENCE FILTER
    # ========================================================
    #
    # Nếu đã có Điều chứa đúng Khoản/Điểm liên quan,
    # không giữ các Điều chỉ match vehicle.
    # ========================================================

    structured_candidates = [
        candidate
        for candidate in candidates
        if candidate.get(
            "relevant_clauses"
        )
    ]

    if structured_candidates:
        candidates = (
            structured_candidates
        )

    return candidates[
        :top_k
    ]


# ============================================================
# DOCUMENT BUILDERS
# ============================================================

def point_to_document(
    article: dict,
    clause: dict,
    point: dict
) -> dict:
    """
    Convert một Điểm pháp luật thành document cho RAG.

    Ngoài nội dung Point, document còn giữ:
        - parent_clause_content
        - article_title

    Mục đích:
        Validator / RAG Chain có thể kiểm tra phần đầu của
        Khoản để xác định mức xử phạt mà không phải retrieve
        lại toàn bộ Điều.

    Lưu ý:
        content vẫn chỉ là Point chính xác để Retriever không
        làm context bị nhiễu.
    """

    article_title = (
        article.get(
            "article_title",
            ""
        )
        or ""
    ).strip()

    point_content = (
        point.get(
            "content",
            ""
        )
        or ""
    ).strip()

    parent_clause_content = (
        clause.get(
            "content",
            ""
        )
        or ""
    ).strip()

    content = (
        f"{article_title}\n"
        f"Khoản {clause['clause_number']}\n"
        f"Điểm {point['point_label']})\n"
        f"{point_content}"
    )

    return {
        "content": content,

        "source": article.get(
            "source"
        ),

        "filename": article.get(
            "filename"
        ),

        "page": point.get(
            "start_page"
        ),

        "score": None,

        "metadata": {
            # ================================================
            # RETRIEVAL TYPE
            # ================================================

            "retrieval_type": (
                "legal_point"
            ),

            # ================================================
            # LEGAL STRUCTURE
            # ================================================

            "article_number": (
                article.get(
                    "article_number"
                )
            ),

            "article_title": (
                article_title
            ),

            "clause_number": (
                clause.get(
                    "clause_number"
                )
            ),

            "point_label": (
                point.get(
                    "point_label"
                )
            ),

            # ================================================
            # IMPORTANT: PARENT CLAUSE
            # ================================================

            "parent_clause_content": (
                parent_clause_content
            ),

            # ================================================
            # ORIGINAL POINT
            # ================================================

            "point_content": (
                point_content
            ),

            # ================================================
            # PAGE
            # ================================================

            "start_page": (
                point.get(
                    "start_page"
                )
            ),

            "end_page": (
                point.get(
                    "end_page"
                )
            ),

            "clause_start_page": (
                clause.get(
                    "start_page"
                )
            ),

            "clause_end_page": (
                clause.get(
                    "end_page"
                )
            ),

            # ================================================
            # SOURCE
            # ================================================

            "original_source": (
                article.get(
                    "original_source"
                )
            ),

            # ================================================
            # RETRIEVAL SCORE
            # ================================================

            "legal_score": (
                article.get(
                    "legal_score"
                )
            ),
        },
    }


def clause_to_document(
    article: dict,
    clause: dict
) -> dict:
    """
    Convert Khoản thành document cho RAG.
    """
    article_title = (
        article.get(
            "article_title",
            ""
        )
        or ""
    ).strip()

    content = (
        f"{article['article_title']}\n"
        f"Khoản {clause['clause_number']}\n"
        f"{clause['content']}"
    )

    return {
        "content": content,

        "source": article[
            "source"
        ],

        "filename": article[
            "filename"
        ],

        "page": clause[
            "start_page"
        ],

        "score": None,

        "metadata": {
            "retrieval_type": (
                "legal_clause"
            ),

            "article_number": (
                article[
                    "article_number"
                ]
            ),

            "clause_number": (
                clause[
                    "clause_number"
                ]
            ),

            "point_label": None,

            "start_page": (
                clause[
                    "start_page"
                ]
            ),

            "end_page": (
                clause[
                    "end_page"
                ]
            ),

            "original_source": (
                article.get(
                    "original_source"
                )
            ),

            "legal_score": (
                article.get(
                    "legal_score"
                )
            ),
        },
    }


def article_to_document(
    article: dict
) -> dict:
    """
    Convert toàn bộ Điều thành document fallback.
    """

    return {
        "content": article[
            "content"
        ],

        "source": article[
            "source"
        ],

        "filename": article[
            "filename"
        ],

        "page": article[
            "start_page"
        ],

        "score": None,

        "metadata": {
            "retrieval_type": (
                "legal_article"
            ),

            "article_number": (
                article[
                    "article_number"
                ]
            ),

            "clause_number": None,

            "point_label": None,

            "start_page": (
                article[
                    "start_page"
                ]
            ),

            "end_page": (
                article[
                    "end_page"
                ]
            ),

            "original_source": (
                article.get(
                    "original_source"
                )
            ),

            "legal_score": (
                article.get(
                    "legal_score"
                )
            ),
        },
    }


# ============================================================
# BUILD STRUCTURED LEGAL DOCUMENTS
# ============================================================

def build_legal_documents(
    articles: list[dict]
) -> list[dict]:
    """
    Build structured evidence.

    Priority:
        Point
        ↓
        Clause
        ↓
        Article fallback
    """

    documents = []

    for article in articles:
        relevant_clauses = (
            article.get(
                "relevant_clauses",
                []
            )
        )

        # ====================================================
        # CLAUSE / POINT
        # ====================================================

        if relevant_clauses:
            for clause in (
                relevant_clauses[
                    :MAX_CLAUSES_PER_ARTICLE
                ]
            ):
                relevant_points = (
                    clause.get(
                        "relevant_points",
                        []
                    )
                )

                if relevant_points:
                    for point in (
                        relevant_points[
                            :MAX_POINTS_PER_CLAUSE
                        ]
                    ):
                        documents.append(
                            point_to_document(
                                article,
                                clause,
                                point
                            )
                        )

                else:
                    documents.append(
                        clause_to_document(
                            article,
                            clause
                        )
                    )

            continue

        # ====================================================
        # ARTICLE FALLBACK
        # ====================================================
        #
        # Chỉ khi Điều match đúng vehicle nhưng OCR/topic
        # matcher không tìm được Khoản/Điểm.
        # ====================================================

        if article.get(
            "vehicle_match"
        ):
            documents.append(
                article_to_document(
                    article
                )
            )

    return documents


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_documents(
    documents: list[dict]
) -> list[dict]:
    """
    Loại document trùng.
    """

    result = []
    seen = set()

    for document in documents:
        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )

        key = (
            document.get(
                "filename"
            ),

            metadata.get(
                "retrieval_type"
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
                document.get(
                    "content",
                    ""
                )
            ),
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
# SEMANTIC FALLBACK FILTER
# ============================================================

def filter_semantic_documents(
    documents: list[dict],
    analysis: dict,
    legal_documents: list[dict]
) -> list[dict]:
    """
    Lọc kết quả semantic.

    Semantic search chỉ đóng vai trò fallback,
    không phải nguồn ưu tiên.
    """

    vehicle = analysis.get(
        "vehicle"
    )

    topic = analysis.get(
        "topic"
    )

    intent = analysis.get(
        "intent"
    )

    filtered = []

    for document in documents:
        filename = document.get(
            "filename",
            ""
        )

        content = document.get(
            "content",
            ""
        )

        # ----------------------------------------------------
        # PENALTY
        # ----------------------------------------------------
        #
        # Câu hỏi xử phạt:
        # ưu tiên NĐ 168 / 238.
        # ----------------------------------------------------

        if (
            intent == "penalty"
            and filename not in {
                DECREE_168,
                DECREE_238,
            }
        ):
            continue

        # ----------------------------------------------------
        # TOPIC
        # ----------------------------------------------------

        if (
            topic is not None
            and not topic_matches_text(
                content,
                topic
            )
        ):
            continue

        # ----------------------------------------------------
        # VEHICLE
        # ----------------------------------------------------

        if vehicle is not None:
            normalized = normalize_text(
                content
            )

            if vehicle == "motorcycle":
                has_motorcycle = (
                    "xe mo to" in normalized
                    or "xe gan may" in normalized
                    or "xe may" in normalized
                )

                has_car = (
                    "xe o to" in normalized
                    or "xe 6 to" in normalized
                )

                if (
                    has_car
                    and not has_motorcycle
                ):
                    continue

            elif vehicle == "car":
                has_car = (
                    "xe o to" in normalized
                    or "xe 6 to" in normalized
                    or "oto" in normalized
                )

                has_motorcycle = (
                    "xe mo to" in normalized
                    or "xe gan may" in normalized
                )

                if (
                    has_motorcycle
                    and not has_car
                ):
                    continue

        filtered.append(
            document
        )

    return filtered[
        :MAX_SEMANTIC_DOCUMENTS
    ]


# ============================================================
# EVIDENCE HELPERS
# ============================================================

def get_retrieval_type(
    document: dict
) -> str:
    """
    Lấy retrieval_type an toàn.
    """

    metadata = (
        document.get(
            "metadata",
            {}
        )
        or {}
    )

    return metadata.get(
        "retrieval_type",
        "semantic"
    )


def has_precise_evidence(
    documents: list[dict]
) -> bool:
    """
    True nếu đã tìm được Point hoặc Clause cụ thể.
    """

    precise_types = {
        "legal_point",
        "legal_clause",
    }

    return any(
        get_retrieval_type(
            document
        )
        in precise_types
        for document in documents
    )


# ============================================================
# FINAL SORTING
# ============================================================

def sort_legal_documents(
    documents: list[dict]
) -> list[dict]:
    """
    Sắp xếp evidence:

        legal_point
        legal_clause
        legal_article
        semantic
    """

    priority = {
        "legal_point": 0,
        "legal_clause": 1,
        "legal_article": 2,
        "semantic": 3,
    }

    def sort_key(
        document: dict
    ):
        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )

        retrieval_type = (
            get_retrieval_type(
                document
            )
        )

        article_number = (
            metadata.get(
                "article_number"
            )
        )

        clause_number = (
            metadata.get(
                "clause_number"
            )
        )

        point_label = (
            metadata.get(
                "point_label"
            )
        )

        return (
            priority.get(
                retrieval_type,
                99
            ),

            article_number
            if isinstance(
                article_number,
                int
            )
            else 999,

            clause_number
            if isinstance(
                clause_number,
                int
            )
            else 999,

            str(
                point_label
                or ""
            ),
        )

    return sorted(
        documents,
        key=sort_key
    )


# ============================================================
# DEBUG
# ============================================================

def print_document_summary(
    documents: list[dict]
) -> None:
    """
    In summary để debug Retriever.
    """

    print(
        "\n========== RETRIEVAL RESULT =========="
    )

    print(
        "Final documents:",
        len(documents)
    )

    for index, document in enumerate(
        documents,
        start=1
    ):
        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
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
            "| Điều:",
            metadata.get(
                "article_number"
            ),
            "| Khoản:",
            metadata.get(
                "clause_number"
            ),
            "| Điểm:",
            metadata.get(
                "point_label"
            ),
            "| Trang:",
            metadata.get(
                "start_page",
                document.get(
                    "page"
                )
            ),
        )


# ============================================================
# MAIN LEGAL RETRIEVAL
# ============================================================

def retrieve_legal_documents(
    question: str,
    top_k: int = 5,
    article_top_k: int = 3
) -> list[dict]:
    """
    Main entry point của Legal Retriever.

    Pipeline:

        Question
            ↓
        Question Analysis
            ↓
        Structured Legal Search
            ↓
        Article
            ↓
        Clause
            ↓
        Point
            ↓
        Có precise evidence?
            ├── YES -> không semantic
            └── NO  -> semantic fallback
            ↓
        Deduplicate
            ↓
        Sort
            ↓
        Return
    """

    print(
        "\n========== LEGAL RETRIEVAL =========="
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
        analysis["vehicle"]
    )

    print(
        "Topic:",
        analysis["topic"]
    )

    print(
        "Intent:",
        analysis["intent"]
    )

    # ========================================================
    # 1. DECREE 238 OVERVIEW
    # ========================================================

    if is_decree_238_overview(
        question
    ):
        print(
            "Mode: DECREE 238 OVERVIEW"
        )

        semantic_candidates = (
            retrieve_documents(
                question=question,
                top_k=max(
                    top_k,
                    8
                )
            )
        )

        documents = [
            document
            for document in semantic_candidates
            if document.get(
                "filename"
            ) == DECREE_238
        ]

        documents = (
            deduplicate_documents(
                documents
            )
        )

        documents = documents[
            :top_k
        ]

        print_document_summary(
            documents
        )

        return documents

    # ========================================================
    # 2. STRUCTURED LEGAL SEARCH
    # ========================================================

    print(
        "Mode: STRUCTURED LEGAL SEARCH"
    )

    articles = search_legal_articles(
        question=question,
        top_k=article_top_k
    )

    print(
        "Legal articles:",
        len(articles)
    )

    for article in articles:
        print(
            "  ->",
            article.get(
                "filename"
            ),
            "| Điều:",
            article.get(
                "article_number"
            ),
            "| score:",
            article.get(
                "legal_score"
            ),
            "| vehicle:",
            article.get(
                "article_vehicle"
            ),
            "| clauses:",
            len(
                article.get(
                    "relevant_clauses",
                    []
                )
            )
        )

    # ========================================================
    # 3. BUILD STRUCTURED EVIDENCE
    # ========================================================

    legal_documents = (
        build_legal_documents(
            articles
        )
    )

    legal_documents = (
        deduplicate_documents(
            legal_documents
        )
    )

    print(
        "Structured documents:",
        len(legal_documents)
    )

    # ========================================================
    # 4. CHECK EVIDENCE QUALITY
    # ========================================================

    precise_evidence = (
        has_precise_evidence(
            legal_documents
        )
    )

    # ========================================================
    # 5. SEMANTIC FALLBACK
    # ========================================================

    semantic_documents = []

    if precise_evidence:
        print(
            "Precise structured evidence found."
        )

        print(
            "Semantic fallback: SKIPPED"
        )

    else:
        print(
            "Precise structured evidence not found."
        )

        print(
            "Running semantic fallback..."
        )

        semantic_candidates = (
            retrieve_documents(
                question=question,
                top_k=top_k
            )
        )

        semantic_documents = (
            filter_semantic_documents(
                documents=semantic_candidates,
                analysis=analysis,
                legal_documents=legal_documents
            )
        )

        semantic_documents = (
            deduplicate_documents(
                semantic_documents
            )
        )

        print(
            "Semantic fallback:",
            len(
                semantic_documents
            )
        )

    # ========================================================
    # 6. COMBINE
    # ========================================================

    combined = (
        legal_documents
        + semantic_documents
    )

    combined = (
        deduplicate_documents(
            combined
        )
    )

    # ========================================================
    # 7. PRIORITY SORT
    # ========================================================

    result = sort_legal_documents(
        combined
    )

    # ========================================================
    # 8. DEBUG SUMMARY
    # ========================================================

    print_document_summary(
        result
    )

    return result