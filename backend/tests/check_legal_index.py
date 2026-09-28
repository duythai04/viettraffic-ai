import json
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INDEX_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "legal_index.json"
)


with open(
    INDEX_PATH,
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


articles = data.get(
    "articles",
    []
)


print(
    "========== LEGAL INDEX CHECK =========="
)

print(
    "Tổng số articles:",
    len(articles)
)


# ============================================================
# GROUP BY DOCUMENT
# ============================================================

documents = defaultdict(list)

for article in articles:

    documents[
        article["filename"]
    ].append(article)


# ============================================================
# CHECK EACH DOCUMENT
# ============================================================

for filename, document_articles in documents.items():

    print()
    print("=" * 70)
    print(filename)
    print("=" * 70)

    numbers = [
        article["article_number"]
        for article in document_articles
    ]

    counter = Counter(numbers)

    duplicates = {
        number: count
        for number, count in counter.items()
        if count > 1
    }

    print(
        "Số article:",
        len(document_articles)
    )

    print(
        "Article number nhỏ nhất:",
        min(numbers) if numbers else None
    )

    print(
        "Article number lớn nhất:",
        max(numbers) if numbers else None
    )

    print(
        "Số article number khác nhau:",
        len(set(numbers))
    )

    print(
        "Duplicate:",
        duplicates
    )


# ============================================================
# DETAIL DUPLICATES
# ============================================================

print()
print(
    "========== DUPLICATE DETAILS =========="
)

for filename, document_articles in documents.items():

    counter = Counter(
        article["article_number"]
        for article in document_articles
    )

    duplicate_numbers = {
        number
        for number, count in counter.items()
        if count > 1
    }

    if not duplicate_numbers:
        continue

    print()
    print(
        f"----- {filename} -----"
    )

    for article in document_articles:

        number = article[
            "article_number"
        ]

        if number not in duplicate_numbers:
            continue

        print()
        print(
            f"Điều {number}"
        )

        print(
            "Trang:",
            article.get("start_page"),
            "-",
            article.get("end_page")
        )

        print(
            "Số khoản:",
            len(
                article.get(
                    "clauses",
                    []
                )
            )
        )

        print(
            "Tiêu đề:"
        )

        print(
            article.get(
                "article_title",
                ""
            )
        )

        content = article.get(
            "content",
            ""
        )

        print(
            "Nội dung đầu:"
        )

        print(
            content[:300]
            .replace(
                "\n",
                " "
            )
        )

        print("-" * 50)