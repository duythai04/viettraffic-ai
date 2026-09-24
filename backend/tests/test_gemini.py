
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


# Đường dẫn tới backend/.env
ENV_PATH = Path(__file__).resolve().parents[1] / ".env"

load_dotenv(ENV_PATH)

api_key = os.getenv("OPENAI_API_KEY")
model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


if not api_key:
    raise ValueError(
        "Không tìm thấy OPENAI_API_KEY trong backend/.env"
    )


def test_openai():

    print("Đang kết nối OpenAI...")
    print(f"Model: {model_name}")

    llm = ChatOpenAI(
        model=model_name,
        api_key=api_key,
        temperature=0
    )

    response = llm.invoke(
        "Xin chào! Hãy giới thiệu ngắn gọn về bạn bằng tiếng Việt."
    )

    print("\n===== OPENAI RESPONSE =====")
    print(response.content)


if __name__ == "__main__":

    try:
        test_openai()

    except Exception as e:
        print("\nKẾT NỐI THẤT BẠI")
        print(type(e).__name__)
        print(str(e))