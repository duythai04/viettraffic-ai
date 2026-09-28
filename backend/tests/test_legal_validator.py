from backend.app.services.rag.legal_retriever import (
    retrieve_legal_documents,
)

from backend.app.services.rag.legal_validator import (
    validate_legal_evidence,
    print_validation_result,
)


QUESTIONS = [
    (
        "Vượt đèn đỏ bằng xe máy "
        "bị xử phạt thế nào?"
    ),

    (
        "Vượt đèn đỏ bằng ô tô "
        "bị xử phạt thế nào?"
    ),

    (
        "Chở trẻ em dưới 10 tuổi trên ô tô "
        "cần tuân thủ quy định gì?"
    ),

    (
        "Xe máy chạy quá tốc độ "
        "bị phạt thế nào?"
    ),
]


def print_separator():
    print(
        "\n"
        + "=" * 100
    )


for question in QUESTIONS:

    print_separator()

    print(
        "QUESTION:"
    )

    print(
        question
    )

    # ========================================================
    # RETRIEVAL
    # ========================================================

    documents = retrieve_legal_documents(
        question=question,
        top_k=5,
        article_top_k=3
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    result = validate_legal_evidence(
        question=question,
        documents=documents
    )

    print_validation_result(
        result
    )

    # ========================================================
    # CONTENT PREVIEW
    # ========================================================

    print(
        "\nEVIDENCE CONTENT:"
    )

    for index, document in enumerate(
        result[
            "valid_documents"
        ],
        start=1
    ):

        content = (
            document.get(
                "content",
                ""
            )
            .replace(
                "\n",
                " "
            )
        )

        print()
        print(
            f"[{index}]"
        )

        print(
            content[:600]
        )


print_separator()

print(
    "TEST COMPLETED"
)