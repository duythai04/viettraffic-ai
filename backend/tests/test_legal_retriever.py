from backend.app.services.rag.legal_retriever import (
    analyze_question,
    search_legal_articles,
    retrieve_legal_documents,
)


QUESTIONS = [
    "Vượt đèn đỏ bằng xe máy bị xử phạt thế nào?",
    "Vượt đèn đỏ bằng ô tô bị xử phạt thế nào?",
    "Chở trẻ em dưới 10 tuổi trên ô tô cần tuân thủ quy định gì?",
    "Xe máy chạy quá tốc độ bị phạt thế nào?",
]


def print_line():
    print(
        "\n"
        + "=" * 90
    )


for question in QUESTIONS:

    print_line()

    print(
        "QUESTION:"
    )

    print(
        question
    )

    # ========================================================
    # QUESTION ANALYSIS
    # ========================================================

    analysis = analyze_question(
        question
    )

    print(
        "\nANALYSIS:"
    )

    print(
        analysis
    )

    # ========================================================
    # ARTICLE SEARCH
    # ========================================================

    articles = search_legal_articles(
        question=question,
        top_k=3
    )

    print(
        "\nARTICLES:"
    )

    for article in articles:

        print(
            f"- {article['filename']}"
            f" | Điều {article['article_number']}"
            f" | score={article['legal_score']}"
            f" | vehicle={article['article_vehicle']}"
        )

        clauses = article.get(
            "relevant_clauses",
            []
        )

        for clause in clauses:

            print(
                f"    Khoản "
                f"{clause['clause_number']}"
                f" | score="
                f"{clause['match_score']}"
            )

            points = clause.get(
                "relevant_points",
                []
            )

            for point in points:

                print(
                    f"        Điểm "
                    f"{point['point_label']}"
                    f" | score="
                    f"{point['match_score']}"
                )

                preview = (
                    point.get(
                        "content",
                        ""
                    )
                    .replace(
                        "\n",
                        " "
                    )
                )

                print(
                    "        ",
                    preview[:250]
                )

    # ========================================================
    # FINAL DOCUMENTS
    # ========================================================

    documents = retrieve_legal_documents(
        question=question,
        top_k=5,
        article_top_k=3
    )

    print(
        "\nFINAL DOCUMENTS:"
    )

    for index, document in enumerate(
        documents,
        start=1
    ):

        metadata = (
            document.get(
                "metadata",
                {}
            )
            or {}
        )

        print()
        print(
            f"[{index}]",
            document.get(
                "filename"
            )
        )

        print(
            "Type:",
            metadata.get(
                "retrieval_type",
                "semantic"
            )
        )

        print(
            "Điều:",
            metadata.get(
                "article_number"
            )
        )

        print(
            "Khoản:",
            metadata.get(
                "clause_number"
            )
        )

        print(
            "Điểm:",
            metadata.get(
                "point_label"
            )
        )

        print(
            "Trang:",
            metadata.get(
                "start_page",
                document.get(
                    "page"
                )
            )
        )

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

        print(
            "Content:",
            content[:400]
        )