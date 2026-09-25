
import sys

from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(
    0,
    str(BACKEND_DIR)
)

from app.services.rag.legal_retriever import (
    search_legal_articles,
    retrieve_legal_documents,
)


QUESTIONS = [
    "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?",
    "Vượt đèn đỏ bằng ô tô bị xử phạt thế nào?",
    "Nghị định 238 sửa đổi những nội dung nào của Nghị định 168?",
]


def test_legal_retriever():

    for question in QUESTIONS:

        print("\n" + "=" * 70)

        print("CÂU HỎI:", question)

        print("=" * 70)

        articles = search_legal_articles(
            question=question,
            top_k=3
        )

        print("\n========== ĐIỀU LUẬT ==========")

        for article in articles:

            print(
                f"\n{article['filename']}"
            )

            print(
                f"Điều {article['article_number']}"
            )

            print(
                article["article_title"]
            )

            print(
                "Trang:",
                article["start_page"],
                "-",
                article["end_page"]
            )

            print(
                "Legal score:",
                article["legal_score"]
            )

            print(
                "\nNội dung mẫu:"
            )

            print(
                article["content"][:800]
            )

        documents = retrieve_legal_documents(
            question=question,
            top_k=5,
            article_top_k=2
        )

        print("\n========== KẾT QUẢ GỘP ==========")

        print(
            "Tổng số tài liệu:",
            len(documents)
        )

        for index, document in enumerate(
            documents[:8],
            start=1
        ):

            metadata = document.get(
                "metadata",
                {}
            )

            print(
                f"{index}. "
                f"{document['filename']} | "
                f"Trang {document['page']} | "
                f"Loại: {metadata.get('retrieval_type', 'semantic')}"
            )


if __name__ == "__main__":

    test_legal_retriever()
