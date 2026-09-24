
import json
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[4]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

OCR_DATA_DIR = BASE_DIR / "data" / "processed" / "ocr"

EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"

# File này lưu đường dẫn database đang được sử dụng.
ACTIVE_DB_CONFIG = EMBEDDINGS_DIR / "active_db.json"

EMBEDDING_MODEL_NAME = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


# ============================================================
# 2. TÌM TÀI LIỆU
# ============================================================

def discover_files(data_dir: Path):

    if not data_dir.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục dữ liệu: {data_dir}"
        )

    files = sorted(
        (
            file
            for file in data_dir.rglob("*")
            if file.is_file()
            and file.suffix.lower() in SUPPORTED_EXTENSIONS
        ),
        key=lambda file: str(file).lower(),
    )

    print("\n========== KIỂM TRA ĐƯỜNG DẪN ==========")
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("OCR_DATA_DIR:", OCR_DATA_DIR)

    print("\n========== FILE ĐƯỢC TÌM THẤY ==========")

    for file in files:
        print("-", file.relative_to(data_dir))

    print(f"\nTổng số file: {len(files)}")

    if not files:
        raise RuntimeError(
            "Không tìm thấy PDF, TXT hoặc DOCX trong data/raw."
        )

    return files


# ============================================================
# 3. CHỌN FILE OCR
# ============================================================

def resolve_document_path(file: Path):

    """
    Nếu file PDF gốc có bản OCR tương ứng thì sử dụng bản OCR.

    Ví dụ:
        data/raw/decrees/238-ndcp.signed.pdf

    Sẽ sử dụng:
        data/processed/ocr/238-ndcp.ocr.pdf

    Nếu chưa có bản OCR, trả về file gốc.
    """

    if file.suffix.lower() != ".pdf":
        return file

    # Chỉ áp dụng quy tắc OCR cho file trong thư mục decrees.
    relative_path = file.relative_to(RAW_DATA_DIR)

    if not relative_path.parts or relative_path.parts[0] != "decrees":
        return file

    filename = file.name

    if filename.endswith(".signed.pdf"):
        ocr_filename = filename.replace(
            ".signed.pdf",
            ".ocr.pdf",
        )
    else:
        ocr_filename = file.stem + ".ocr.pdf"

    ocr_file = OCR_DATA_DIR / ocr_filename

    if ocr_file.exists():
        print(f"  -> Sử dụng bản OCR: {ocr_file.name}")
        return ocr_file

    print(f"  -> Chưa có bản OCR: {ocr_filename}")
    print("  -> Thử đọc PDF gốc.")

    return file


# ============================================================
# 4. ĐỌC TÀI LIỆU
# ============================================================

def load_documents(data_dir: Path):

    files = discover_files(data_dir)

    documents = []

    for file in files:

        relative_path = file.relative_to(data_dir)

        print(f"\nĐang đọc: {relative_path}")

        try:

            actual_file = resolve_document_path(file)

            extension = actual_file.suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(str(actual_file))

            elif extension == ".txt":
                loader = TextLoader(
                    str(actual_file),
                    encoding="utf-8",
                )

            elif extension == ".docx":
                loader = Docx2txtLoader(str(actual_file))

            else:
                raise RuntimeError(
                    f"Định dạng không hỗ trợ: {extension}"
                )

            loaded_docs = loader.load()

            if not loaded_docs:
                raise RuntimeError(
                    "File không trả về trang/tài liệu nào."
                )

            # Loại bỏ các trang không có nội dung.
            valid_docs = [
                doc
                for doc in loaded_docs
                if doc.page_content.strip()
            ]

            empty_count = len(loaded_docs) - len(valid_docs)

            print(
                f"  -> Tổng số trang/tài liệu: {len(loaded_docs)}"
            )

            print(
                f"  -> Có nội dung: {len(valid_docs)}"
            )

            print(
                f"  -> Rỗng: {empty_count}"
            )

            # Không được âm thầm bỏ qua một tài liệu hoàn toàn rỗng.
            if not valid_docs:
                raise RuntimeError(
                    "Tài liệu không có văn bản để embedding. "
                    "Nếu là PDF scan, cần OCR trước."
                )

            # Chuẩn hóa metadata.
            for doc in valid_docs:

                # source trỏ đến file thực tế được đọc.
                doc.metadata["source"] = str(actual_file)

                # Giữ tên tài liệu gốc để hiển thị nguồn.
                doc.metadata["filename"] = file.name

                doc.metadata["relative_path"] = (
                    relative_path.as_posix()
                )

                doc.metadata["document_type"] = (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else "other"
                )

                doc.metadata["is_ocr"] = (
                    actual_file != file
                )

                doc.metadata["original_source"] = str(file)

            documents.extend(valid_docs)

            print(
                f"  -> THÀNH CÔNG: {len(valid_docs)} "
                "trang/tài liệu có nội dung"
            )

        except Exception as error:

            print(
                f"  -> LỖI: {type(error).__name__}"
            )

            print(
                f"  -> Chi tiết: {error}"
            )

            raise RuntimeError(
                f"Không thể đọc tài liệu: {relative_path}"
            ) from error

    print("\n========== KẾT QUẢ ĐỌC TÀI LIỆU ==========")

    print(
        "Tổng số trang/tài liệu có nội dung:",
        len(documents),
    )

    return documents


# ============================================================
# 5. CHIA CHUNKS
# ============================================================

def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=[
            "\nĐiều ",
            "\nKhoản ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )

    chunks = text_splitter.split_documents(documents)

    # Bảo đảm không lưu chunk rỗng.
    chunks = [
        chunk
        for chunk in chunks
        if chunk.page_content.strip()
    ]

    if not chunks:
        raise RuntimeError(
            "Không tạo được chunk nào từ tài liệu."
        )

    print("\n========== KẾT QUẢ CHIA CHUNKS ==========")

    print("Tổng số chunks:", len(chunks))

    counts = Counter(
        chunk.metadata.get("filename", "Không rõ")
        for chunk in chunks
    )

    for filename, count in sorted(counts.items()):
        print(f"- {filename}: {count} chunks")

    return chunks


# ============================================================
# 6. TẠO VECTOR DATABASE
# ============================================================

def build_vector_store():

    print("\n========================================")
    print("       VIETTRAFFIC AI - INGESTION")
    print("========================================")

    # Bước 1: Đọc toàn bộ tài liệu.
    documents = load_documents(RAW_DATA_DIR)

    # Bước 2: Chia chunks.
    chunks = split_documents(documents)

    # Bước 3: Khởi tạo embedding model.
    print("\nĐang khởi tạo embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
    )

    # Bước 4: Tạo đường dẫn database mới.
    EMBEDDINGS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )

    new_db_dir = EMBEDDINGS_DIR / f"chroma_db_{timestamp}"

    print("\nĐang tạo ChromaDB mới...")
    print("Đường dẫn:", new_db_dir)

    # Bước 5: Tạo database mới, không đụng vào DB cũ.
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(new_db_dir),
    )

    # Bước 6: Kiểm tra số lượng chunks.
    saved_count = vector_db._collection.count()

    print("Số chunks đã lưu:", saved_count)

    if saved_count != len(chunks):
        raise RuntimeError(
            "Số chunks trong ChromaDB không khớp "
            "với số chunks đã tạo."
        )

    # Bước 7: Cập nhật đường dẫn DB đang sử dụng.
    # Chỉ thực hiện sau khi DB mới được tạo thành công.
    config = {
        "persist_directory": str(new_db_dir),
        "embedding_model": EMBEDDING_MODEL_NAME,
        "chunk_count": saved_count,
        "created_at": datetime.now().isoformat(),
    }

    temp_config = EMBEDDINGS_DIR / "active_db.json.tmp"

    try:

        temp_config.write_text(
            json.dumps(
                config,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Thay thế file cấu hình, không đổi tên thư mục SQLite.
        temp_config.replace(ACTIVE_DB_CONFIG)

    except Exception:

        print(
            "Không thể cập nhật active_db.json. "
            "Database cũ vẫn được giữ nguyên."
        )

        raise

    print("\n========================================")
    print("        INGESTION THÀNH CÔNG")
    print("========================================")

    print("Tổng số chunks:", saved_count)
    print("ChromaDB:", new_db_dir)
    print("Cấu hình:", ACTIVE_DB_CONFIG)

    print(
        "\nLưu ý: Nếu FastAPI đang chạy, hãy khởi động lại "
        "để retriever nạp database mới."
    )


if __name__ == "__main__":
    build_vector_store()