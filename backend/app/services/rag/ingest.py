import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"

EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

def load_documents(data_dir: Path):
    documents = []
    if not data_dir.exists():
        print(f"Thư mục {data_dir} không tồn tại.")
        return documents

    for file in data_dir.glob("**/*"):
        if file.suffix.lower() == ".pdf":
            loader = PyPDFLoader(str(file))
            documents.extend(loader.load())
        elif file.suffix == ".txt":
            loader = TextLoader(str(file))
            documents.extend(loader.load())
        elif file.suffix == "docx":
            loader = Docx2txtLoader(str(file))
            documents.extend(loader.load())
    print(f"Đã tải {len(documents)} trang/tài liệu từ {data_dir}")
    return documents

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap = 120, 
        separators=["\nĐiều ", "\nKhoản ", "\n\n", "\n", " ", ""]
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Đã chia thành {len(chunks)} đoạn nhỏ (chunks)")
    return chunks

def build_vector_store():

    # load tài liệu
    documents = load_documents(RAW_DATA_DIR)
    if not documents:
        print("không có tài liệu để xử lý.")
        return

    # chia nhỏ tài liệu
    chunks = split_documents(documents)

    # khởi tạo embeddings
    embeddings =HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs = {"device": "cpu"}
    )
    # tạo vector store
    vector_db = Chroma.from_documents(
        documents = chunks, 
        embedding = embeddings, 
        persist_directory = str(VECTOR_DB_DIR)
    )

    print("--- HOÀN TẤT! Vector DB đã sẵn sàng sử dụng ---")

if __name__ == "__main__":
    build_vector_store()