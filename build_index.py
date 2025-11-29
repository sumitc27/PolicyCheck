import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DOCS_PATH = "docs"
INDEX_PATH_BASE = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def create_vector_dbs():
    """
    Scans the DOCS_PATH for PDFs.
    Creates a separate FAISS index for EACH document.
    Saves them in subfolders under INDEX_PATH_BASE.
    """
    if not os.path.exists(INDEX_PATH_BASE):
        os.makedirs(INDEX_PATH_BASE)
        print(f"Created directory: {INDEX_PATH_BASE}")

    if not os.path.exists(DOCS_PATH):
        os.makedirs(DOCS_PATH)
        print(f"Created directory: {DOCS_PATH}. Please put your PDFs here!")
        return

    pdf_files = [f for f in os.listdir(DOCS_PATH) if f.lower().endswith(".pdf")]
    
    if not pdf_files:
        print("No PDF documents found in 'docs/'. Please add some files.")
        return

    print(f"Found {len(pdf_files)} documents to process.")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    for pdf_file in pdf_files:
        try:
            index_name = os.path.splitext(pdf_file)[0].lower().replace(" ", "_")
            full_index_path = os.path.join(INDEX_PATH_BASE, index_name)
            
            if os.path.exists(full_index_path):
                print(f"Skipping '{pdf_file}' - Index already exists at {full_index_path}")
                continue

            print(f"Processing: {pdf_file}...")
            
            file_path = os.path.join(DOCS_PATH, pdf_file)
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            for doc in documents:
                doc.metadata['source'] = pdf_file

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            chunks = text_splitter.split_documents(documents)
            print(f"  - Split into {len(chunks)} chunks.")

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(full_index_path)
            print(f"  - Saved index to: {full_index_path}")
        
        except Exception as e:
            print(f"ERROR processing {pdf_file}: {e}")

    print("\nAll processing complete.")

if __name__ == "__main__":
    create_vector_dbs()