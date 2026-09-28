import json
import re
import unicodedata

from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

RAW_DIR = BASE_DIR / "data" / "raw"
OCR_DIR = BASE_DIR / "data" / "processed" / "ocr"

INDEX_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "legal_index.json"
)


DECREE_168 = "168-nd-cp.signed.pdf"
DECREE_238 = "238-ndcp.signed.pdf"


# ============================================================
# TEXT NORMALIZATION
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa text phục vụ nhận diện cấu trúc.

    Ví dụ:
        Điều 7  -> dieu 7
        Điểm đ) -> diem d)

    Không dùng hàm này để thay đổi nội dung pháp luật được lưu.
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

    text = text.replace("đ", "d")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# STRUCTURE PATTERNS
# ============================================================

ARTICLE_PATTERN = re.compile(
    r"^\s*dieu\s+(\d{1,3})\s*[.:]?",
    re.IGNORECASE
)


CLAUSE_PATTERN = re.compile(
    r"^\s*(\d{1,2})\s*[.]\s+(.+)$"
)


POINT_PATTERN = re.compile(
    r"^\s*([a-zđ])\s*[\)\.]\s+(.+)$",
    re.IGNORECASE
)


# ============================================================
# STRUCTURE DETECTION
# ============================================================

def detect_article(line: str) -> int | None:
    """
    Nhận diện TIÊU ĐỀ Điều luật thật.

    Ví dụ hợp lệ:
        Điều 7. Xử phạt...
        Điều 24. Công trình an toàn giao thông đường bộ
        ĐIỀU 10. Quy tắc chung

    Loại các dòng tham chiếu:
        Điều 24 của Luật này.
        Điều 26 của Nghị định này;
        Điều 32 (điểm c khoản 17)...
        Điều 15, Điều 16, Điều 17...
        Điều 21, điểm b khoản 9a Điều 32...
    """

    if not line:
        return None

    original = re.sub(
        r"\s+",
        " ",
        line.strip()
    )

    normalized = normalize_text(
        original
    )

    # ========================================================
    # 1. Phải bắt đầu bằng "Điều <số>"
    # ========================================================

    match = re.match(
        r"^dieu\s+(\d{1,3})(.*)$",
        normalized,
        re.IGNORECASE
    )

    if not match:
        return None

    try:
        article_number = int(
            match.group(1)
        )
    except ValueError:
        return None

    remainder = (
        match.group(2)
        or ""
    ).strip()

    # Không có nội dung sau số Điều.
    if not remainder:
        return None

    # ========================================================
    # 2. LOẠI THAM CHIẾU "Điều X của ..."
    # ========================================================

    if re.match(
        r"^\s*cua\s+",
        remainder
    ):
        return None

    # ========================================================
    # 3. LOẠI "Điều X (điểm..., khoản...)"
    # ========================================================

    if re.match(
        r"^\s*\(",
        remainder
    ):
        return None

    # ========================================================
    # 4. LOẠI DANH SÁCH:
    #
    # Điều 15, Điều 16, Điều 17...
    # Điều 21, điểm b khoản 9a Điều 32...
    # ========================================================

    if remainder.startswith(","):
        return None

    # ========================================================
    # 5. TIÊU ĐỀ THẬT NÊN CÓ DẤU "." HOẶC ":"
    #    NGAY SAU SỐ ĐIỀU
    #
    # Điều 24. ...
    # Điều 7: ...
    #
    # OCR đôi khi đọc:
    # Diéu 18. ...
    # normalize_text vẫn xử lý được.
    # ========================================================

    if not re.match(
        r"^\s*[.:]",
        remainder
    ):
        return None

    # ========================================================
    # 6. LẤY PHẦN SAU "." / ":"
    # ========================================================

    title_text = re.sub(
        r"^\s*[.:]\s*",
        "",
        remainder
    ).strip()

    if not title_text:
        return None

    # ========================================================
    # 7. LOẠI MỘT SỐ THAM CHIẾU CÒN SÓT
    # ========================================================

    reference_patterns = [
        r"^cua\s+luat\s+nay",
        r"^cua\s+nghi\s+dinh\s+nay",
        r"^cua\s+van\s+ban\s+nay",
    ]

    if any(
        re.match(pattern, title_text)
        for pattern in reference_patterns
    ):
        return None

    return article_number
    """
    Nhận diện:

        Điều 7.
        Điều 7:
        ĐIỀU 7

    Sau khi normalize:

        dieu 7.
    """

    normalized = normalize_text(line)

    match = ARTICLE_PATTERN.match(
        normalized
    )

    if not match:
        return None

    try:
        return int(
            match.group(1)
        )
    except ValueError:
        return None


def detect_clause(line: str) -> int | None:
    """
    Nhận diện khoản:

        1. Nội dung...
        2. Nội dung...

    Không nhận:
        3.8 ...
        1.35 m
    """

    if not line:
        return None

    line = line.strip()

    match = CLAUSE_PATTERN.match(
        line
    )

    if not match:
        return None

    try:
        clause_number = int(
            match.group(1)
        )
    except ValueError:
        return None

    # Khoản trong văn bản pháp luật thường không quá lớn.
    # Giới hạn này giúp giảm một số false positive từ OCR.
    if clause_number <= 0 or clause_number > 50:
        return None

    return clause_number


def detect_point(line: str) -> str | None:
    """
    Nhận diện điểm:

        a) ...
        b) ...
        đ) ...

    Đồng thời hỗ trợ OCR / PDF dạng:

        a. ...
        b. ...
    """

    if not line:
        return None

    line = line.strip()

    match = POINT_PATTERN.match(
        line
    )

    if not match:
        return None

    return match.group(1).lower()


# ============================================================
# PDF DISCOVERY
# ============================================================

def resolve_source_path(
    pdf_path: Path
) -> Path:
    """
    Chọn file thực tế để parser đọc.

    Nghị định 168/238:
        đọc bản OCR.

    Các tài liệu khác:
        đọc PDF gốc.
    """

    if pdf_path.name == DECREE_168:

        return (
            OCR_DIR
            / "168-nd-cp.ocr.pdf"
        )

    if pdf_path.name == DECREE_238:

        return (
            OCR_DIR
            / "238-ndcp.ocr.pdf"
        )

    return pdf_path


def find_pdf_files() -> list[dict]:
    """
    Tìm tất cả PDF trong data/raw.

    filename:
        tên văn bản gốc.

    source_path:
        file thực tế parser đọc.

    original_path:
        file gốc.
    """

    files = []

    if not RAW_DIR.exists():

        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {RAW_DIR}"
        )

    for pdf_path in sorted(
        RAW_DIR.rglob("*.pdf")
    ):

        source_path = resolve_source_path(
            pdf_path
        )

        if not source_path.exists():

            print(
                "[WARNING] Không tìm thấy file để đọc:"
            )

            print(
                f"          {source_path}"
            )

            continue

        files.append({
            "filename": pdf_path.name,
            "source_path": str(source_path),
            "original_path": str(pdf_path),
        })

    return files


# ============================================================
# ARTICLE STRUCTURE PARSER
# ============================================================

def create_clause(
    clause_number: int,
    line: str,
    page: int
) -> dict:

    return {
        "clause_number": clause_number,
        "start_page": page,
        "end_page": page,
        "_lines": [line],
        "points": [],
    }


def create_point(
    point_label: str,
    line: str,
    page: int
) -> dict:

    return {
        "point_label": point_label,
        "start_page": page,
        "end_page": page,
        "_lines": [line],
    }


def finalize_point(
    point: dict
) -> dict:

    point["content"] = "\n".join(
        point.pop(
            "_lines",
            []
        )
    ).strip()

    return point


def finalize_clause(
    clause: dict
) -> dict:

    clause["content"] = "\n".join(
        clause.pop(
            "_lines",
            []
        )
    ).strip()

    clause["points"] = [
        finalize_point(point)
        for point in clause.get(
            "points",
            []
        )
    ]

    return clause


def parse_article_structure(
    article: dict
) -> dict:
    """
    Chuyển:

        Article
            raw lines

    thành:

        Article
            clauses
                points

    QUAN TRỌNG:

    Nội dung của điểm vẫn được giữ trong content của khoản.

    Nhờ vậy:
        clause["content"]

    chứa đầy đủ:

        3. Phạt tiền...
        a) ...
        b) ...
        c) ...

    Điều này rất quan trọng cho Legal Retriever.
    """

    lines = article.pop(
        "_lines",
        []
    )

    clauses = []

    current_clause = None
    current_point = None

    # lines[0] thường là tiêu đề Điều.
    for line_info in lines[1:]:

        line = (
            line_info.get(
                "text",
                ""
            )
            or ""
        ).strip()

        page = line_info.get(
            "page"
        )

        if not line:
            continue

        clause_number = detect_clause(
            line
        )

        point_label = detect_point(
            line
        )

        # ====================================================
        # NEW CLAUSE
        # ====================================================

        if clause_number is not None:

            current_clause = create_clause(
                clause_number=clause_number,
                line=line,
                page=page
            )

            clauses.append(
                current_clause
            )

            current_point = None

            continue

        # ====================================================
        # NEW POINT
        # ====================================================

        if (
            point_label is not None
            and current_clause is not None
        ):

            current_point = create_point(
                point_label=point_label,
                line=line,
                page=page
            )

            current_clause[
                "points"
            ].append(
                current_point
            )

            # Điểm là một phần của khoản.
            current_clause[
                "_lines"
            ].append(
                line
            )

            current_clause[
                "end_page"
            ] = page

            continue

        # ====================================================
        # CONTINUATION
        # ====================================================

        if current_clause is not None:

            current_clause[
                "_lines"
            ].append(
                line
            )

            current_clause[
                "end_page"
            ] = page

        if current_point is not None:

            current_point[
                "_lines"
            ].append(
                line
            )

            current_point[
                "end_page"
            ] = page

    # ========================================================
    # FINALIZE CLAUSES
    # ========================================================

    article["clauses"] = [
        finalize_clause(clause)
        for clause in clauses
    ]

    # Nội dung đầy đủ của Điều.
    article["content"] = "\n".join(
        (
            line_info.get(
                "text",
                ""
            )
            or ""
        ).strip()
        for line_info in lines
        if (
            line_info.get(
                "text",
                ""
            )
            or ""
        ).strip()
    ).strip()

    return article


# ============================================================
# PARSE PDF
# ============================================================

def parse_pdf_articles(
    filename: str,
    source_path: str,
    original_path: str
) -> list[dict]:

    print(
        f"\nĐang phân tích: {filename}"
    )

    print(
        f"Nguồn đọc: {source_path}"
    )

    pages = PyPDFLoader(
        source_path
    ).load()

    print(
        f"Số trang: {len(pages)}"
    )

    articles = []

    current_article = None

    def finalize_current_article():

        nonlocal current_article

        if current_article is None:
            return

        parsed_article = (
            parse_article_structure(
                current_article
            )
        )

        articles.append(
            parsed_article
        )

        current_article = None

    for page_index, page in enumerate(
        pages
    ):

        page_number = (
            page_index + 1
        )

        page_content = (
            page.page_content
            or ""
        )

        lines = (
            page_content
            .splitlines()
        )

        for raw_line in lines:

            line = raw_line.strip()

            if not line:
                continue

            article_number = detect_article(
                line
            )

            # =================================================
            # NEW ARTICLE
            # =================================================

            if article_number is not None:

                finalize_current_article()

                current_article = {
                    "filename": filename,

                    "source": source_path,

                    "original_source": (
                        original_path
                    ),

                    "article_number": (
                        article_number
                    ),

                    "article_title": (
                        line
                    ),

                    "start_page": (
                        page_number
                    ),

                    "end_page": (
                        page_number
                    ),

                    "_lines": [],
                }

            # =================================================
            # ARTICLE CONTENT
            # =================================================

            if current_article is not None:

                current_article[
                    "_lines"
                ].append({
                    "text": line,
                    "page": page_number,
                })

                current_article[
                    "end_page"
                ] = page_number

    finalize_current_article()

    clause_count = sum(
        len(
            article.get(
                "clauses",
                []
            )
        )
        for article in articles
    )

    point_count = sum(
        len(
            clause.get(
                "points",
                []
            )
        )
        for article in articles
        for clause in article.get(
            "clauses",
            []
        )
    )

    print(
        "Số điều nhận diện:",
        len(articles)
    )

    print(
        "Số khoản nhận diện:",
        clause_count
    )

    print(
        "Số điểm nhận diện:",
        point_count
    )

    return articles


# ============================================================
# INDEX VALIDATION
# ============================================================

def validate_articles(
    articles: list[dict]
) -> list[str]:
    """
    Kiểm tra một số lỗi cấu trúc cơ bản.

    Đây không phải kiểm tra tính đúng đắn pháp lý.
    """

    warnings = []

    for article in articles:

        filename = article.get(
            "filename",
            "unknown"
        )

        article_number = article.get(
            "article_number"
        )

        if not article.get(
            "content"
        ):

            warnings.append(
                f"{filename} - Điều "
                f"{article_number}: "
                "không có content."
            )

        clauses = article.get(
            "clauses",
            []
        )

        # Một điều không có khoản chưa chắc là lỗi.
        # Chỉ ghi nhận để kiểm tra.
        if not clauses:

            warnings.append(
                f"{filename} - Điều "
                f"{article_number}: "
                "không nhận diện được khoản."
            )

        for clause in clauses:

            if not clause.get(
                "content"
            ):

                warnings.append(
                    f"{filename} - Điều "
                    f"{article_number} - Khoản "
                    f"{clause.get('clause_number')}: "
                    "content rỗng."
                )

    return warnings


# ============================================================
# BUILD LEGAL INDEX
# ============================================================

def build_legal_index() -> dict:

    print(
        "========== BUILD LEGAL INDEX =========="
    )

    pdf_files = find_pdf_files()

    if not pdf_files:

        raise RuntimeError(
            "Không tìm thấy PDF hợp lệ để xây legal index."
        )

    print(
        "Số tài liệu:",
        len(pdf_files)
    )

    all_articles = []

    document_stats = []

    for file_info in pdf_files:

        articles = parse_pdf_articles(
            filename=file_info[
                "filename"
            ],

            source_path=file_info[
                "source_path"
            ],

            original_path=file_info[
                "original_path"
            ],
        )

        all_articles.extend(
            articles
        )

        clause_count = sum(
            len(
                article.get(
                    "clauses",
                    []
                )
            )
            for article in articles
        )

        point_count = sum(
            len(
                clause.get(
                    "points",
                    []
                )
            )
            for article in articles
            for clause in article.get(
                "clauses",
                []
            )
        )

        document_stats.append({
            "filename": file_info[
                "filename"
            ],

            "article_count": len(
                articles
            ),

            "clause_count": (
                clause_count
            ),

            "point_count": (
                point_count
            ),
        })

    warnings = validate_articles(
        all_articles
    )

    total_clauses = sum(
        len(
            article.get(
                "clauses",
                []
            )
        )
        for article in all_articles
    )

    total_points = sum(
        len(
            clause.get(
                "points",
                []
            )
        )
        for article in all_articles
        for clause in article.get(
            "clauses",
            []
        )
    )

    index_data = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),

        "total_documents": len(
            pdf_files
        ),

        "total_articles": len(
            all_articles
        ),

        "total_clauses": (
            total_clauses
        ),

        "total_points": (
            total_points
        ),

        "documents": pdf_files,

        "document_stats": (
            document_stats
        ),

        "warnings": warnings,

        "articles": all_articles,
    }

    INDEX_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        INDEX_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            index_data,
            file,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n========== LEGAL INDEX =========="
    )

    print(
        "File:",
        INDEX_PATH
    )

    print(
        "Tổng tài liệu:",
        len(pdf_files)
    )

    print(
        "Tổng số điều:",
        len(all_articles)
    )

    print(
        "Tổng số khoản:",
        total_clauses
    )

    print(
        "Tổng số điểm:",
        total_points
    )

    print(
        "\n========== DOCUMENT STATS =========="
    )

    for stats in document_stats:

        print(
            f"{stats['filename']}: "
            f"{stats['article_count']} điều | "
            f"{stats['clause_count']} khoản | "
            f"{stats['point_count']} điểm"
        )

    if warnings:

        print(
            "\n========== WARNINGS =========="
        )

        for warning in warnings[:20]:

            print(
                "-",
                warning
            )

        if len(warnings) > 20:

            print(
                f"... và {len(warnings) - 20} warning khác."
            )

    print(
        "\nBuild legal index hoàn tất."
    )

    return index_data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    build_legal_index()