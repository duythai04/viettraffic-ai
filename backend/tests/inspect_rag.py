
import sys
from pathlib import Path
from collections import Counter

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag.retriever import (
    get_vector_store,
    retrieve_documents,
)


def inspect_database():
    vector_db = get_vector_store()

    collection = vector_db._collection

    data = collection.get(
        include=["metadatas"]
    )

    metadata_list = data.get("metadatas", [])

    counts = Counter(
        metadata.get("filename", "Không rõ")
        for metadata in metadata_list
        if metadata
    )

    print("\n========== SỐ CHUNKS THEO TÀI LIỆU ==========")

    for filename, count in counts.items():
        print(f"{filename}: {count}")

    print("\nTỔNG:", sum(counts.values()))


def inspect_question(question, top_k=10):
    print("\n" + "=" * 70)
    print("CÂU HỎI:", question)
    print("=" * 70)

    results = retrieve_documents(
        question=question,
        top_k=top_k
    )

    for index, doc in enumerate(results, start=1):
        print(f"\n========== CHUNK {index} ==========")
        print("Tài liệu:", doc["filename"])
        print("Trang:", doc["page"])
        print("Distance:", doc["score"])
        print("\nNỘI DUNG:")
        print(doc["content"])


def inspect_decree_238():
    vector_db = get_vector_store()

    print("\n========== KIỂM TRA NGHỊ ĐỊNH 238 ==========")

    data = vector_db._collection.get(
        where={
            "filename": "238-ndcp.signed.pdf"
        },
        include=["documents", "metadatas"],
        limit=3
    )

    print("Số chunks mẫu:", len(data["ids"]))

    for index, content in enumerate(
        data["documents"],
        start=1
    ):
        print(f"\n--- MẪU {index} ---")
        print("Metadata:", data["metadatas"][index - 1])
        print("Nội dung:")
        print(content[:1500])


if __name__ == "__main__":
    inspect_database()

    inspect_decree_238()

    inspect_question(
        "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?",
        top_k=10
    )

    inspect_question(
        "Nghị định 238 sửa đổi những nội dung nào của Nghị định 168?",
        top_k=10
    )