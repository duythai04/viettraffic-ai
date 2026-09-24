
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_DB_DIR = (
    BASE_DIR / "data" / "embeddings" / "chroma_db"
)

EMBEDDING_MODEL_NAME = (
    "bkai-foundation-models/vietnamese-bi-encoder"
)


embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs={"device": "cpu"}
)

vector_db = Chroma(
    persist_directory=str(VECTOR_DB_DIR),
    embedding_function=embeddings
)

collection = vector_db.get(
    include=["metadatas"]
)

print("\n===== CHROMADB =====")

print("Tổng số chunks:", len(collection["ids"]))

sources = {}

for metadata in collection["metadatas"]:

    source = metadata.get("source", "Không rõ")

    filename = Path(source).name

    sources[filename] = sources.get(filename, 0) + 1


print("\n===== DANH SÁCH TÀI LIỆU =====")

for filename, count in sorted(sources.items()):

    print(f"{filename}: {count} chunks")