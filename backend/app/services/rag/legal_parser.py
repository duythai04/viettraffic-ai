
import json
import re
import unicodedata

from datetime import datetime, timezone
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader


BASE_DIR = Path(__file__).resolve().parents[4]

RAW_DIR = BASE_DIR / "data" / "raw"

OCR_DIR = BASE_DIR / "data" / "processed" / "ocr"

INDEX_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "legal_index.json"
)


# ============================================================
# NORMALIZE
# ============================================================

def normalize_text(text: str) -> str:

    text = unicodedata.normalize(
        "NFD",
        text.lower()
    )

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.replace("đ", "d")

    return re.sub(r"\s+", " ", text).strip()


# ============================================================
# PATTERNS
# ============================================================

ARTICLE_PATTERN = re.compile(
    r"^\s*dieu\s+(\d{1,3})\s*[.:]",
    re.IGNORECASE
)

CLAUSE_PATTERN = re.compile(
    r"^\s*(\d{1,2})\s*[.]\s+(.+)"
)

POINT_PATTERN = re.compile(
    r"^\s*([a-zđ])\s*[)]\s+(.+)",
    re.IGNORECASE
)


def detect_article(line: str):

    normalized = normalize_text(line)

    match = ARTICLE_PATTERN.match(normalized)

    if not match:
        return None

    return int(match.group(1))


def detect_clause(line: str):

    match = CLAUSE_PATTERN.match(line)

    if not match:
        return None

    return int(match.group(1))


def detect_point(line: str):

    match = POINT_PATTERN.match(line)

    if not match:
        return None

    return match.group(1).lower()


# ============================================================
# PDF FILES
# ============================================================

def find_pdf_files():

    files = []

    for pdf_path in RAW_DIR.rglob("*.pdf"):

        if pdf_path.name == "168-nd-cp.signed.pdf":

            source_path = (
                OCR_DIR / "168-nd-cp.ocr.pdf"
            )

        elif pdf_path.name == "238-ndcp.signed.pdf":

            source_path = (
                OCR_DIR / "238-ndcp.ocr.pdf"
            )

        else:

            source_path = pdf_path

        if not source_path.exists():

            print(
                f"Không tìm thấy: {source_path}"
            )

            continue

        files.append({
            "filename": pdf_path.name,
            "source_path": str(source_path),
            "original_path": str(pdf_path),
        })

    return files


# ============================================================
# BUILD CLAUSES AND POINTS
# ============================================================


def parse_article_structure(article: dict):

    lines = article.pop("_lines")

    clauses = []

    current_clause = None
    current_point = None

    for line_info in lines[1:]:

        line = line_info["text"]
        page = line_info["page"]

        clause_number = detect_clause(line)
        point_label = detect_point(line)

        # ========================================
        # NEW CLAUSE
        # ========================================

        if clause_number is not None:

            current_clause = {
                "clause_number": clause_number,
                "start_page": page,
                "end_page": page,
                "lines": [line],
                "points": [],
            }

            clauses.append(current_clause)

            current_point = None

            continue

        # ========================================
        # NEW POINT
        # ========================================

        if (
            point_label is not None
            and current_clause is not None
        ):

            current_point = {
                "point_label": point_label,
                "start_page": page,
                "end_page": page,
                "lines": [line],
            }

            current_clause["points"].append(
                current_point
            )

            # QUAN TRỌNG:
            # Điểm cũng thuộc nội dung khoản.
            current_clause["lines"].append(line)

            current_clause["end_page"] = page

            continue

        # ========================================
        # CONTINUE CONTENT
        # ========================================

        if current_clause is not None:

            current_clause["lines"].append(line)

            current_clause["end_page"] = page

        if current_point is not None:

            current_point["lines"].append(line)

            current_point["end_page"] = page

    # ========================================
    # FINALIZE
    # ========================================

    for clause in clauses:

        clause["content"] = "\n".join(
            clause.pop("lines")
        ).strip()

        for point in clause["points"]:

            point["content"] = "\n".join(
                point.pop("lines")
            ).strip()

    article["content"] = "\n".join(
        item["text"]
        for item in lines
    ).strip()

    article["clauses"] = clauses

    return article


    lines = article.pop("_lines")

    clauses = []

    current_clause = None

    current_point = None

    for line_info in lines[1:]:

        line = line_info["text"]

        page = line_info["page"]

        clause_number = detect_clause(line)

        point_label = detect_point(line)

        # --------------------------
        # NEW CLAUSE
        # --------------------------

        if clause_number is not None:

            current_point = None

            current_clause = {
                "clause_number": clause_number,
                "start_page": page,
                "end_page": page,
                "lines": [line],
                "points": [],
            }

            clauses.append(current_clause)

            continue

        # --------------------------
        # NEW POINT
        # --------------------------

        if (
            point_label is not None
            and current_clause is not None
        ):

            current_point = {
                "point_label": point_label,
                "start_page": page,
                "end_page": page,
                "lines": [line],
            }

            current_clause["points"].append(
                current_point
            )

            current_clause["end_page"] = page

            continue

        # --------------------------
        # CONTINUE CONTENT
        # --------------------------

        if current_point is not None:

            current_point["lines"].append(line)

            current_point["end_page"] = page

        if current_clause is not None:

            current_clause["lines"].append(line)

            current_clause["end_page"] = page

    # --------------------------
    # FINALIZE
    # --------------------------

    for clause in clauses:

        clause["content"] = "\n".join(
            clause.pop("lines")
        ).strip()

        for point in clause["points"]:

            point["content"] = "\n".join(
                point.pop("lines")
            ).strip()

    article["content"] = "\n".join(
        item["text"]
        for item in lines
    ).strip()

    article["clauses"] = clauses

    return article


# ============================================================
# PARSE PDF
# ============================================================

def parse_pdf_articles(
    filename: str,
    source_path: str,
    original_path: str
):

    print(f"\nĐang phân tích: {filename}")

    pages = PyPDFLoader(source_path).load()

    articles = []

    current_article = None

    def finalize_current():

        nonlocal current_article

        if current_article is None:
            return

        articles.append(
            parse_article_structure(
                current_article
            )
        )

        current_article = None

    for page_index, page in enumerate(pages):

        page_number = page_index + 1

        lines = (
            page.page_content or ""
        ).splitlines()

        for line in lines:

            article_number = detect_article(line)

            if article_number is not None:

                finalize_current()

                current_article = {
                    "filename": filename,
                    "source": source_path,
                    "original_source": original_path,
                    "article_number": article_number,
                    "article_title": line.strip(),
                    "start_page": page_number,
                    "end_page": page_number,
                    "_lines": [],
                }

            if current_article is not None:

                current_article["_lines"].append({
                    "text": line,
                    "page": page_number,
                })

                current_article["end_page"] = (
                    page_number
                )

    finalize_current()

    print(
        "Số điều nhận diện:",
        len(articles)
    )

    print(
        "Số khoản nhận diện:",
        sum(
            len(article["clauses"])
            for article in articles
        )
    )

    return articles


# ============================================================
# BUILD INDEX
# ============================================================

def build_legal_index():

    pdf_files = find_pdf_files()

    all_articles = []

    for file_info in pdf_files:

        articles = parse_pdf_articles(
            filename=file_info["filename"],
            source_path=file_info["source_path"],
            original_path=file_info["original_path"],
        )

        all_articles.extend(articles)

    index_data = {
        "created_at": datetime.now(
            timezone.utc
        ).isoformat(),
        "total_articles": len(all_articles),
        "documents": pdf_files,
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

    print("\n========== LEGAL INDEX ==========")

    print("File:", INDEX_PATH)

    print(
        "Tổng số điều:",
        len(all_articles)
    )

    print(
        "Tổng số khoản:",
        sum(
            len(article["clauses"])
            for article in all_articles
        )
    )

    return index_data


if __name__ == "__main__":

    build_legal_index()
