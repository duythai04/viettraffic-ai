
from pathlib import Path
from functools import lru_cache

import torch

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Đường dẫn tới thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parents[4]

VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"

EMBEDDING_MODEL_NAME = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)


@lru_cache(maxsize=1)
def get_vector_store():

    if not VECTOR_DB_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy ChromaDB tại: {VECTOR_DB_DIR}. "
            "Hãy chạy ingest.py trước."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Embedding device: {device}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": device}
    )

    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings
    )

    return vector_db


def retrieve_documents(question: str, top_k: int = 5):

    if not question.strip():
        return []

    vector_db = get_vector_store()

    results = vector_db.similarity_search_with_score(
        query=question,
        k=top_k
    )

    documents = []

    for doc, score in results:

        documents.append({
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Không rõ"),
            "page": doc.metadata.get("page"),
            "score": float(score),
            "metadata": doc.metadata
        })

    return documents


if __name__ == "__main__":

    question = "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?"

    results = retrieve_documents(question)

    for index, doc in enumerate(results, start=1):

        print(f"\n===== KẾT QUẢ {index} =====")
        print("Nguồn:", doc["source"])
        print("Trang:", doc["page"])
        print("Distance:", doc["score"])
        print(doc["content"])