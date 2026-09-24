
import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import (
    SystemMessage,
    HumanMessage
)

from .retriever import retrieve_documents
from .prompts import SYSTEM_PROMPT, build_user_prompt


# Đường dẫn tới backend/.env
BASE_DIR = Path(__file__).resolve().parents[4]
ENV_PATH = BASE_DIR / "backend" / ".env"

load_dotenv(ENV_PATH)


@lru_cache(maxsize=1)
def get_llm():

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise ValueError(
            "Không tìm thấy OPENAI_API_KEY trong backend/.env"
        )

    model_name = os.getenv(
        "OPENAI_MODEL",
        "gpt-4o-mini"
    )

    return ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0
    )


def format_context(documents):

    context_parts = []

    for index, doc in enumerate(documents, start=1):

        source = doc.get("source", "Không rõ")
        page = doc.get("page")

        page_display = (
            page + 1
            if isinstance(page, int)
            else "Không rõ"
        )

        context_parts.append(
            f"""
[TÀI LIỆU {index}]

Nguồn: {source}
Trang PDF: {page_display}

Nội dung:
{doc["content"]}
"""
        )

    return "\n\n".join(context_parts)


def build_sources(documents):

    sources = []
    seen = set()

    for doc in documents:

        source = doc.get("source")
        page = doc.get("page")

        key = (source, page)

        if key in seen:
            continue

        seen.add(key)

        sources.append({
            "source": source,
            "page": (
                page + 1
                if isinstance(page, int)
                else None
            )
        })

    return sources


def ask_viettraffic(question: str):

    question = question.strip()

    if not question:
        return {
            "answer": "Vui lòng nhập câu hỏi.",
            "sources": []
        }

    # BƯỚC 1: Retrieval
    documents = retrieve_documents(
        question=question,
        top_k=5
    )

    if not documents:
        return {
            "answer": (
                "Tôi chưa tìm thấy tài liệu pháp luật "
                "phù hợp trong cơ sở dữ liệu."
            ),
            "sources": []
        }

    # BƯỚC 2: Tạo context
    context = format_context(documents)

    # BƯỚC 3: Tạo prompt
    user_prompt = build_user_prompt(
        question=question,
        context=context
    )

    # BƯỚC 4: Gọi GPT-4o mini
    llm = get_llm()

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt)
    ])

    # BƯỚC 5: Lấy câu trả lời
    answer = response.content

    # BƯỚC 6: Lấy nguồn tài liệu
    sources = build_sources(documents)

    return {
        "question": question,
        "answer": answer,
        "sources": sources
    }


if __name__ == "__main__":

    question = input(
        "\nNhập câu hỏi về luật giao thông: "
    )

    result = ask_viettraffic(question)

    print("\n========== VIETTRAFFIC AI ==========\n")

    print(result["answer"])

    print("\n========== TÀI LIỆU TRUY XUẤT ==========\n")

    for source in result["sources"]:
        print(source)