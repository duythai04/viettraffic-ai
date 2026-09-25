
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
    "Quy định về việc chở trẻ em trên ô tô là gì?",
]


def run_test():

    for question in QUESTIONS:

        print("\n" + "=" * 75)

        print("CÂU HỎI:", question)

        print("=" * 75)

        articles = search_legal_articles(
            question=question,
            top_k=3
        )

        print("\n========== ARTICLES ==========")

        for article in articles:

            print(
                "\nVăn bản:",
                article["filename"]
            )

            print(
                "Điều:",
                article["article_number"]
            )

            print(
                "Tiêu đề:",
                article["article_title"]
            )

            print(
                "Trang:",
                article["start_page"],
                "-",
                article["end_page"]
            )

            print(
                "Điểm tìm kiếm:",
                article["legal_score"]
            )

            clauses = article.get(
                "relevant_clauses",
                []
            )

            print(
                "Số khoản phù hợp:",
                len(clauses)
            )

            for clause in clauses[:2]:

                print(
                    "\n--- KHOẢN",
                    clause["clause_number"],
                    "---"
                )

                print(
                    "Trang:",
                    clause["start_page"],
                    "-",
                    clause["end_page"]
                )

                print(
                    clause["content"][:1500]
                )

        documents = retrieve_legal_documents(
            question=question,
            top_k=5,
            article_top_k=2
        )

        print("\n========== DOCUMENTS ==========")

        for document in documents[:10]:

            metadata = document.get(
                "metadata",
                {}
            )

            print(
                document["filename"],
                "| Trang:",
                document["page"],
                "| Loại:",
                metadata.get(
                    "retrieval_type",
                    "semantic"
                ),
                "| Điều:",
                metadata.get(
                    "article_number"
                ),
                "| Khoản:",
                metadata.get(
                    "clause_number"
                )
            )


if __name__ == "__main__":

    run_test()
