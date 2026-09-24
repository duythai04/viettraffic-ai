
import json
import re
import unicodedata

from pathlib import Path
from functools import lru_cache

import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

ACTIVE_DB_CONFIG = (
    BASE_DIR / "data" / "embeddings" / "active_db.json"
)

DEFAULT_EMBEDDING_MODEL = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)

DECREE_168 = "168-nd-cp.signed.pdf"
DECREE_238 = "238-ndcp.signed.pdf"


# ============================================================
# 2. UTILITIES
# ============================================================

def normalize_text(text: str) -> str:
    """
    Chuyển về chữ thường, bỏ dấu tiếng Việt.
    Chỉ dùng để nhận diện ý định, không sửa nội dung pháp luật.
    """

    text = unicodedata.normalize("NFD", text.lower())

    text = "".join(
        char
        for char in text
        if unicodedata.category(char) != "Mn"
    )

    text = text.replace("đ", "d")

    return re.sub(r"\s+", " ", text).strip()


def get_active_db_config():

    if not ACTIVE_DB_CONFIG.exists():
        raise FileNotFoundError(
            f"Không tìm thấy: {ACTIVE_DB_CONFIG}. "
            "Hãy chạy ingest.py trước."
        )

    with open(
        ACTIVE_DB_CONFIG,
        "r",
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    db_directory = config.get("persist_directory")

    if not db_directory:
        raise RuntimeError(
            "active_db.json thiếu persist_directory."
        )

    db_path = Path(db_directory)

    if not db_path.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy ChromaDB: {db_path}"
        )

    return {
        "persist_directory": str(db_path),
        "embedding_model": config.get(
            "embedding_model",
            DEFAULT_EMBEDDING_MODEL
        ),
        "chunk_count": config.get("chunk_count"),
    }


# ============================================================
# 3. EMBEDDINGS + VECTOR STORE
# ============================================================

@lru_cache(maxsize=2)
def get_embeddings(model_name: str):

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Embedding device: {device}")
    print(f"Embedding model: {model_name}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )


@lru_cache(maxsize=2)
def _load_vector_store(
    persist_directory: str,
    embedding_model: str
):

    print("\n========== KHỞI TẠO RETRIEVER ==========")
    print("ChromaDB:", persist_directory)

    vector_db = Chroma(
        persist_directory=persist_directory,
        embedding_function=get_embeddings(embedding_model)
    )

    count = vector_db._collection.count()

    print("Số chunks trong database:", count)

    if count == 0:
        raise RuntimeError(
            "ChromaDB đang trống. Hãy kiểm tra ingestion."
        )

    return vector_db


def get_vector_store():

    config = get_active_db_config()

    return _load_vector_store(
        config["persist_directory"],
        config["embedding_model"]
    )


# ============================================================
# 4. QUESTION ANALYSIS
# ============================================================

def analyze_question(question: str) -> dict:

    normalized = normalize_text(question)

    mentions_168 = bool(
        re.search(r"\b168\b", normalized)
    )

    mentions_238 = bool(
        re.search(r"\b238\b", normalized)
    )

    amendment_keywords = [
        "sua doi",
        "bo sung",
        "thay doi",
        "diem moi",
        "khac nhau",
        "so sanh",
    ]

    asks_amendments = any(
        keyword in normalized
        for keyword in amendment_keywords
    )

    # Chỉ nhận diện khi người dùng nêu rõ loại phương tiện.
    # Tránh tự suy diễn "xe máy" thành "xe máy chuyên dùng".
    mentions_special_vehicle = any(
        keyword in normalized
        for keyword in [
            "xe may chuyen dung",
            "may keo",
            "xe may thi cong",
        ]
    )

    mentions_motorcycle = (
        not mentions_special_vehicle
        and any(
            keyword in normalized
            for keyword in [
                "xe may",
                "xe mo to",
                "xe gan may",
            ]
        )
    )

    mentions_car = any(
        keyword in normalized
        for keyword in [
            "o to",
            "xe hoi",
        ]
    )

    asks_penalty = any(
        keyword in normalized
        for keyword in [
            "phat",
            "xu phat",
            "muc tien",
            "tru diem",
        ]
    )

    # Câu hỏi dạng tổng quan về nội dung sửa đổi.
    # Cần đọc toàn bộ văn bản sửa đổi, không chỉ top 5 chunks.
    asks_238_overview = (
        mentions_238
        and asks_amendments
        and not asks_penalty
    )

    return {
        "mentions_168": mentions_168,
        "mentions_238": mentions_238,
        "asks_amendments": asks_amendments,
        "asks_238_overview": asks_238_overview,
        "mentions_motorcycle": mentions_motorcycle,
        "mentions_special_vehicle": mentions_special_vehicle,
        "mentions_car": mentions_car,
        "asks_penalty": asks_penalty,
    }


def build_search_queries(
    question: str,
    analysis: dict
) -> list[str]:

    queries = [question]

    normalized = normalize_text(question)

    if analysis["mentions_motorcycle"]:

        if (
            "den do" in normalized
            or "den tin hieu" in normalized
        ):
            queries.append(
                "Người điều khiển xe mô tô xe gắn máy "
                "không chấp hành hiệu lệnh của đèn tín hiệu "
                "giao thông mức phạt tiền"
            )

    if analysis["mentions_special_vehicle"]:

        queries.append(
            "Xử phạt người điều khiển xe máy chuyên dùng "
            "vi phạm quy tắc giao thông đường bộ"
        )

    if analysis["mentions_car"] and analysis["asks_penalty"]:

        queries.append(
            "Xử phạt người điều khiển xe ô tô "
            "vi phạm quy tắc giao thông đường bộ"
        )

    return list(dict.fromkeys(queries))


# ============================================================
# 5. CONVERT CHROMA RESULTS
# ============================================================

def convert_document(
    doc,
    distance=None
) -> dict:

    metadata = dict(doc.metadata or {})

    page_index = metadata.get("page")

    page_number = (
        page_index + 1
        if isinstance(page_index, int)
        else None
    )

    return {
        "content": doc.page_content,
        "source": metadata.get("source", "Không rõ"),
        "filename": metadata.get("filename", "Không rõ"),
        "page": page_number,
        "page_index": page_index,
        "score": (
            float(distance)
            if distance is not None
            else None
        ),
        "metadata": metadata,
    }


def document_key(document: dict) -> tuple:

    return (
        document["filename"],
        document["page_index"],
        document["content"],
    )


def deduplicate_documents(documents: list) -> list:

    unique = []
    seen = set()

    for document in documents:

        key = document_key(document)

        if key in seen:
            continue

        seen.add(key)
        unique.append(document)

    return unique


# ============================================================
# 6. SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question: str,
    top_k: int = 5,
    filename: str | None = None
) -> list[dict]:

    vector_db = get_vector_store()

    search_filter = (
        {"filename": filename}
        if filename
        else None
    )

    results = vector_db.similarity_search_with_score(
        query=question,
        k=top_k,
        filter=search_filter
    )

    return [
        convert_document(doc, distance)
        for doc, distance in results
    ]


# ============================================================
# 7. LẤY CHUNKS THEO METADATA
# ============================================================

def get_documents_by_filename(
    filename: str
) -> list[dict]:
    """
    Lấy toàn bộ chunks của một tài liệu.
    Dùng cho câu hỏi tổng quan về Nghị định 238.
    """

    vector_db = get_vector_store()

    data = vector_db._collection.get(
        where={"filename": filename},
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        data.get("documents", []),
        data.get("metadatas", [])
    ):

        metadata = metadata or {}

        page_index = metadata.get("page")

        documents.append({
            "content": content or "",
            "source": metadata.get("source", "Không rõ"),
            "filename": metadata.get("filename", filename),
            "page": (
                page_index + 1
                if isinstance(page_index, int)
                else None
            ),
            "page_index": page_index,
            "score": None,
            "metadata": metadata,
        })

    documents.sort(
        key=lambda item: (
            item["page_index"]
            if isinstance(item["page_index"], int)
            else 999999
        )
    )

    return documents


def get_page_documents(
    filename: str,
    page_indexes: list[int]
) -> list[dict]:
    """
    Lấy chunks ở các trang chỉ định.
    Không thực hiện embedding hoặc semantic search.
    """

    page_indexes = sorted(set(
        page
        for page in page_indexes
        if isinstance(page, int) and page >= 0
    ))

    if not page_indexes:
        return []

    vector_db = get_vector_store()

    data = vector_db._collection.get(
        where={
            "$and": [
                {"filename": filename},
                {"page": {"$in": page_indexes}},
            ]
        },
        include=["documents", "metadatas"]
    )

    documents = []

    for content, metadata in zip(
        data.get("documents", []),
        data.get("metadatas", [])
    ):

        metadata = metadata or {}

        page_index = metadata.get("page")

        documents.append({
            "content": content or "",
            "source": metadata.get("source", "Không rõ"),
            "filename": metadata.get("filename", filename),
            "page": (
                page_index + 1
                if isinstance(page_index, int)
                else None
            ),
            "page_index": page_index,
            "score": None,
            "metadata": metadata,
        })

    documents.sort(
        key=lambda item: (
            item["page_index"]
            if isinstance(item["page_index"], int)
            else 999999
        )
    )

    return documents


# ============================================================
# 8. MỞ RỘNG NGỮ CẢNH THEO TRANG
# ============================================================

def expand_page_context(
    documents: list[dict],
    max_seed_documents: int = 3
) -> list[dict]:
    """
    Lấy toàn bộ chunks trên trang chứa kết quả tìm kiếm
    và trang liền trước/sau.

    Mục đích: hạn chế tách hành vi vi phạm khỏi
    tiêu đề khoản và mức tiền phạt.

    Đây là mở rộng ngữ cảnh, không phải xác minh pháp lý.
    """

    expanded = list(documents)

    seeds = documents[:max_seed_documents]

    pages_by_filename = {}

    for document in seeds:

        filename = document.get("filename")
        page_index = document.get("page_index")

        if (
            not filename
            or not isinstance(page_index, int)
        ):
            continue

        pages = pages_by_filename.setdefault(
            filename,
            set()
        )

        pages.update([
            page_index - 1,
            page_index,
            page_index + 1,
        ])

    for filename, pages in pages_by_filename.items():

        expanded.extend(
            get_page_documents(
                filename,
                list(pages)
            )
        )

    return deduplicate_documents(expanded)


# ============================================================
# 9. LEGAL RETRIEVAL
# ============================================================

def retrieve_documents(
    question: str,
    top_k: int = 5
) -> list[dict]:

    if not question or not question.strip():
        return []

    if top_k < 1:
        raise ValueError("top_k phải >= 1.")

    question = question.strip()

    analysis = analyze_question(question)

    # --------------------------------------------------------
    # CASE A: Hỏi tổng quan về Nghị định 238
    # --------------------------------------------------------

    if analysis["asks_238_overview"]:

        print(
            "Retrieval mode: FULL DECREE 238"
        )

        return get_documents_by_filename(
            DECREE_238
        )

    # --------------------------------------------------------
    # CASE B: Hỏi rõ về Nghị định 238
    # --------------------------------------------------------

    if analysis["mentions_238"]:

        print(
            "Retrieval mode: TARGETED DECREE 238"
        )

        documents = semantic_search(
            question=question,
            top_k=top_k,
            filename=DECREE_238
        )

        # Nếu người dùng đồng thời nhắc Nghị định 168,
        # lấy thêm dữ liệu từ văn bản gốc.
        if analysis["mentions_168"]:

            documents.extend(
                semantic_search(
                    question=question,
                    top_k=max(2, top_k // 2),
                    filename=DECREE_168
                )
            )

        documents = deduplicate_documents(documents)

        return expand_page_context(
            documents,
            max_seed_documents=2
        )

    # --------------------------------------------------------
    # CASE C: Hỏi rõ về Nghị định 168
    # --------------------------------------------------------

    if analysis["mentions_168"]:

        print(
            "Retrieval mode: TARGETED DECREE 168"
        )

        documents = semantic_search(
            question=question,
            top_k=top_k,
            filename=DECREE_168
        )

        # Kiểm tra thêm văn bản sửa đổi.
        documents.extend(
            semantic_search(
                question=question,
                top_k=2,
                filename=DECREE_238
            )
        )

        documents = deduplicate_documents(documents)

        return expand_page_context(
            documents,
            max_seed_documents=3
        )

    # --------------------------------------------------------
    # CASE D: Câu hỏi chung
    # --------------------------------------------------------

    print(
        "Retrieval mode: HYBRID LEGAL SEARCH"
    )

    queries = build_search_queries(
        question,
        analysis
    )

    documents = []

    # Tìm kiếm toàn database.
    for query in queries:

        documents.extend(
            semantic_search(
                question=query,
                top_k=top_k
            )
        )

    # Với câu hỏi xử phạt, ưu tiên bổ sung Nghị định 168.
    if analysis["asks_penalty"]:

        for query in queries:

            documents.extend(
                semantic_search(
                    question=query,
                    top_k=top_k,
                    filename=DECREE_168
                )
            )

        # Lấy thêm Nghị định 238 để kiểm tra
        # xem có nội dung sửa đổi liên quan không.
        documents.extend(
            semantic_search(
                question=question,
                top_k=2,
                filename=DECREE_238
            )
        )

    documents = deduplicate_documents(documents)

    return expand_page_context(
        documents,
        max_seed_documents=3
    )


# ============================================================
# 10. TEST
# ============================================================

if __name__ == "__main__":

    questions = [
        "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?",
        "Nghị định 238 sửa đổi những nội dung nào của Nghị định 168?",
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print("CÂU HỎI:", question)

        results = retrieve_documents(
            question,
            top_k=5
        )

        print("Số chunks:", len(results))

        for index, document in enumerate(
            results,
            start=1
        ):

            print(
                f"{index}. "
                f"{document['filename']} | "
                f"Trang {document['page']} | "
                f"Distance {document['score']}"
            )