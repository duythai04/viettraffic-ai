
import sys
from pathlib import Path


# Thêm backend vào Python path
BACKEND_DIR = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(BACKEND_DIR))


from app.services.rag.pipeline import ask_viettraffic


def main():

    print("\n================================")
    print("       VIETTRAFFIC AI")
    print("================================")

    while True:

        question = input(
            "\nBạn: "
        ).strip()

        if question.lower() in [
            "exit",
            "quit",
            "thoat"
        ]:
            print("Đã thoát VietTraffic AI.")
            break

        if not question:
            continue

        try:

            result = ask_viettraffic(question)

            print("\nVietTraffic AI:")
            print(result["answer"])

            print("\nNguồn tài liệu truy xuất:")

            for source in result["sources"]:

                print(
                    f"- {source['source']} "
                    f"(Trang {source['page']})"
                )

        except Exception as e:

            print("\nCó lỗi xảy ra:")
            print(type(e).__name__)
            print(str(e))


if __name__ == "__main__":
    main()