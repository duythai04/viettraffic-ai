
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


BASE_DIR = Path(__file__).resolve().parents[2]

PDF_DIR = BASE_DIR / "data" / "raw" / "decrees"


def check_pdf(pdf_path: Path):

    print("\n" + "=" * 70)
    print("FILE:", pdf_path.name)

    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()

    print("Tổng số trang:", len(documents))

    total_characters = 0
    pages_with_text = 0

    for index, doc in enumerate(documents, start=1):

        content = doc.page_content.strip()

        total_characters += len(content)

        if content:
            pages_with_text += 1

        # Chỉ hiển thị 3 trang đầu
        if index <= 3:

            print(f"\n--- TRANG {index} ---")
            print("Số ký tự:", len(content))
            print("Nội dung mẫu:")
            print(repr(content[:500]))

    print("\n===== TỔNG KẾT =====")
    print("Số trang có văn bản:", pages_with_text)
    print("Tổng số ký tự:", total_characters)


if __name__ == "__main__":

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print("Không tìm thấy file PDF trong decrees.")

    for pdf_file in pdf_files:
        check_pdf(pdf_file)