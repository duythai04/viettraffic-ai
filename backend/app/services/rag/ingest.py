
import shutil
from collections import Counter
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma




BASE_DIR = Path(__file__).resolve().parents[4]

RAW_DATA_DIR = BASE_DIR / "data" / "raw"

VECTOR_DB_DIR = (
    BASE_DIR / "data" / "embeddings" / "chroma_db"
)

# Database tạm để tránh xóa database cũ trước khi ingest thành công
TEMP_VECTOR_DB_DIR = (
    BASE_DIR / "data" / "embeddings" / "chroma_db_building"
)

BACKUP_VECTOR_DB_DIR = (
    BASE_DIR / "data" / "embeddings" / "chroma_db_backup"
)

EMBEDDING_MODEL_NAME = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}



#  TÌM TÀI LIỆU


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
        key=lambda file: str(file).lower()
    )

    print("\n========== KIỂM TRA ĐƯỜNG DẪN ==========")
    print("BASE_DIR:", BASE_DIR)
    print("RAW_DATA_DIR:", RAW_DATA_DIR)
    print("VECTOR_DB_DIR:", VECTOR_DB_DIR)

    print("\n========== FILE ĐƯỢC TÌM THẤY ==========")

    for file in files:
        print("-", file.relative_to(data_dir))

    print(f"\nTổng số file: {len(files)}")

    if not files:
        raise RuntimeError(
            "Không tìm thấy PDF, TXT hoặc DOCX trong data/raw."
        )

    return files



#  ĐỌC TÀI LIỆU


def load_documents(data_dir: Path):

    files = discover_files(data_dir)

    documents = []

    for file in files:

        relative_path = file.relative_to(data_dir)

        print(f"\nĐang đọc: {relative_path}")

        try:

            extension = file.suffix.lower()

            if extension == ".pdf":
                loader = PyPDFLoader(str(file))

            elif extension == ".txt":
                loader = TextLoader(
                    str(file),
                    encoding="utf-8"
                )

            elif extension == ".docx":
                loader = Docx2txtLoader(str(file))

            loaded_docs = loader.load()

            if not loaded_docs:
                raise RuntimeError(
                    "File không trả về trang/tài liệu nào."
                )

            # Chuẩn hóa metadata để dễ lọc nguồn về sau
            for doc in loaded_docs:

                doc.metadata["source"] = str(file)

                doc.metadata["filename"] = file.name

                doc.metadata["relative_path"] = (
                    relative_path.as_posix()
                )

                doc.metadata["document_type"] = (
                    relative_path.parts[0]
                    if len(relative_path.parts) > 1
                    else "other"
                )

            documents.extend(loaded_docs)

            print(
                f"  -> THÀNH CÔNG: {len(loaded_docs)} trang/tài liệu"
            )

        except Exception as error:

            print(f"  -> LỖI: {type(error).__name__}")
            print(f"  -> Chi tiết: {error}")

            # Không tiếp tục xây DB nếu thiếu bất kỳ file nào
            raise RuntimeError(
                f"Không thể đọc tài liệu: {relative_path}"
            ) from error

    print("\n========== KẾT QUẢ ĐỌC TÀI LIỆU ==========")
    print("Tổng số trang/tài liệu:", len(documents))

    return documents



#  CHIA CHUNKS


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
            ""
        ]
    )

    chunks = text_splitter.split_documents(documents)

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



#  TẠO VECTOR DATABASE


def build_vector_store():

    print("\n========================================")
    print("       VIETTRAFFIC AI - INGESTION")
    print("========================================")

    # Bước 1: Đọc toàn bộ tài liệu
    documents = load_documents(RAW_DATA_DIR)

    # Bước 2: Chia chunks
    chunks = split_documents(documents)

    # Bước 3: Khởi tạo embedding model
    print("\nĐang khởi tạo embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"}
    )

    # Bước 4: Dọn database tạm của lần chạy trước
    if TEMP_VECTOR_DB_DIR.exists():
        shutil.rmtree(TEMP_VECTOR_DB_DIR)

    TEMP_VECTOR_DB_DIR.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("\nĐang tạo ChromaDB mới...")

    # Bước 5: Tạo database ở thư mục tạm
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(TEMP_VECTOR_DB_DIR)
    )

    # Bước 6: Kiểm tra số lượng chunks đã lưu
    saved_count = vector_db._collection.count()

    print("Số chunks đã lưu:", saved_count)

    if saved_count != len(chunks):
        raise RuntimeError(
            "Số chunks trong ChromaDB không khớp "
            "với số chunks đã tạo."
        )

    # Giải phóng kết nối trước khi đổi thư mục trên Windows
    del vector_db

    # Bước 7: Sao lưu database cũ
    if BACKUP_VECTOR_DB_DIR.exists():
        shutil.rmtree(BACKUP_VECTOR_DB_DIR)

    if VECTOR_DB_DIR.exists():
        VECTOR_DB_DIR.rename(BACKUP_VECTOR_DB_DIR)

    # Bước 8: Đưa database mới vào vị trí chính thức
    try:

        TEMP_VECTOR_DB_DIR.rename(VECTOR_DB_DIR)

    except Exception:

        # Khôi phục database cũ nếu thay thế thất bại
        if (
            not VECTOR_DB_DIR.exists()
            and BACKUP_VECTOR_DB_DIR.exists()
        ):
            BACKUP_VECTOR_DB_DIR.rename(VECTOR_DB_DIR)

        raise

    print("\n========================================")
    print("        INGESTION THÀNH CÔNG")
    print("========================================")

    print("Tổng số chunks:", saved_count)
    print("ChromaDB:", VECTOR_DB_DIR)

    if BACKUP_VECTOR_DB_DIR.exists():
        print("Bản sao lưu DB cũ:", BACKUP_VECTOR_DB_DIR)


if __name__ == "__main__":
    build_vector_store()