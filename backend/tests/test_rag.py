
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_DIR))

from app.services.rag.rag_chain import ask_rag


def test_rag():

    print("\n========================================")
    print("         VIETTRAFFIC AI - RAG TEST")
    print("========================================")

    questions = [
        "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?",
        "Nghị định 238 sửa đổi những nội dung nào của Nghị định 168?",
        "Quy định về việc chở trẻ em trên ô tô là gì?",
    ]

    for index, question in enumerate(questions, start=1):

        print("\n" + "=" * 65)
        print(f"CÂU HỎI {index}: {question}")
        print("=" * 65)

        try:
            result = ask_rag(
                question=question,
                top_k=5
            )

        except Exception as error:

            print("\nLỖI KHI CHẠY RAG:")
            print(type(error).__name__, str(error))

            continue

        print("\n========== CÂU TRẢ LỜI ==========")
        print(result["answer"])

        print("\n========== NGUỒN ĐƯỢC LLM TRÍCH DẪN ==========")

        cited_sources = result.get("sources", [])

        if not cited_sources:
            print("Không có nguồn được trích dẫn.")

        for source in cited_sources:

            print(
                f"[{source.get('id')}] "
                f"{source.get('filename')} "
                f"| Trang {source.get('page')}"
            )

        print("\n========== NGUỒN ĐƯA VÀO CONTEXT ==========")

        retrieved_sources = result.get(
            "retrieved_sources",
            []
        )

        for source in retrieved_sources:

            distance = source.get("distance")

            distance_display = (
                f"{distance:.4f}"
                if isinstance(distance, (int, float))
                else "N/A"
            )

            print(
                f"[{source.get('id')}] "
                f"{source.get('filename')} "
                f"| Trang {source.get('page')} "
                f"| Distance: {distance_display}"
            )

        print("\n========== THỐNG KÊ ==========")

        print(
            "Số chunks truy xuất:",
            result.get("retrieved_count", 0)
        )

        print(
            "Số chunks đưa vào context:",
            len(retrieved_sources)
        )

        print(
            "Số nguồn được trích dẫn:",
            len(cited_sources)
        )

        usage = result.get("usage")

        if usage:

            print(
                "Input tokens:",
                usage.get("prompt_tokens")
            )

            print(
                "Output tokens:",
                usage.get("completion_tokens")
            )

            print(
                "Total tokens:",
                usage.get("total_tokens")
            )


if __name__ == "__main__":
    test_rag()