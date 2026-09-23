from pathlib import Path
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Đường dẫn đến Vector DB đã lưu
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent
VECTOR_DB_DIR = BASE_DIR / "data" / "embeddings" / "chroma_db"
EMBEDDING_MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder"

def query_rag(question: str, top_k: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Tải Embeddings Model
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': device}
    )
    
    # 2. Kết nối tới ChromaDB đã được tạo
    vector_db = Chroma(
        persist_directory=str(VECTOR_DB_DIR),
        embedding_function=embeddings
    )
    
    # 3. Tìm kiếm k kết quả liên quan nhất
    results = vector_db.similarity_search(question, k=top_k)
    
    print(f"\n================ CÂU HỎI: '{question}' ================\n")
    if not results:
        print("Không tìm thấy đoạn văn bản luật nào phù hợp trong Vector DB!")
        return

    for i, doc in enumerate(results, 1):
        source = doc.metadata.get('source', 'Không rõ')
        page = doc.metadata.get('page', 'N/A')
        print(f"--- [Kết quả {i}] | Nguồn: {source} (Trang {page}) ---")
        print(doc.page_content.strip())
        print("-" * 60 + "\n")

if __name__ == "__main__":
    query_rag("Mức phạt vượt đèn đỏ xe máy là bao nhiêu?")