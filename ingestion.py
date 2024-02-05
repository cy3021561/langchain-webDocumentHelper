import os
from dotenv import load_dotenv
from langchain.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from consts import PINECONE_INDEX
from pinecone import Pinecone as PC

load_dotenv()
pc = PC(api_key=os.environ.get("PINECONE_API_KEY"))


def ingest_docs() -> None:
    custom_html_tag = ("main", {})
    loader = ReadTheDocsLoader(
        path="./langchain-docs/python.langchain.com",
        encoding="ISO-8859-1",
        custom_html_tag=custom_html_tag,
    )
    raw_docments = loader.load()
    print(f"Loaded {len(raw_docments)} documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(documents=raw_docments)
    print(f"Splitted into {len(documents)} chunks.")

    for doc in documents:
        new_url = doc.metadata["source"]
        print(new_url)
        new_url = new_url.replace("\\", "/")
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})
        print(new_url)

    print(f"Going to insert {len(documents)} to Pinecone.")
    embeddings = OpenAIEmbeddings()
    Pinecone.from_documents(
        documents=documents, embedding=embeddings, index_name=PINECONE_INDEX
    )
    print("****** Added to Pinecone vector space ******")


if __name__ == "__main__":
    ingest_docs()
