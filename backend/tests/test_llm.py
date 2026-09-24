
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


# ==================================================
# 1. ĐỌC FILE .ENV
# ==================================================

BASE_DIR = Path(__file__).resolve().parents[2]

ENV_PATH = BASE_DIR / "backend" / ".env"

load_dotenv(ENV_PATH)

API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "Không tìm thấy OPENAI_API_KEY trong backend/.env"
    )


# ==================================================
# 2. KHỞI TẠO OPENAI CLIENT
# ==================================================

client = OpenAI(api_key=API_KEY)


# ==================================================
# 3. TEST LLM
# ==================================================

def test_llm():

    print("\n====================================")
    print("       VIETTRAFFIC AI - LLM TEST")
    print("====================================")

    question = "Xin chào! Bạn có thể làm gì?"

    print("\nCâu hỏi:", question)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Bạn là VietTraffic AI, trợ lý hỗ trợ "
                    "tra cứu thông tin pháp luật giao thông "
                    "đường bộ Việt Nam. "
                    "Trả lời bằng tiếng Việt, rõ ràng, dễ hiểu. "
                    "Không tự bịa quy định pháp luật."
                )
            },
            {
                "role": "user",
                "content": question
            }
        ],
        temperature=0.2,
        max_tokens=300
    )

    answer = response.choices[0].message.content

    print("\n========== CÂU TRẢ LỜI ==========")
    print(answer)

    print("\n========== THÔNG TIN ==========")
    print("Model:", response.model)
    print("Input tokens:", response.usage.prompt_tokens)
    print("Output tokens:", response.usage.completion_tokens)
    print("Total tokens:", response.usage.total_tokens)

    print("\nTEST LLM THÀNH CÔNG!")


if __name__ == "__main__":
    test_llm()