import os
import glob
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.documents import Document

load_dotenv()

INDEX_PATH_BASE = "vector_store"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "gemini-1.5-flash-latest"

class ClaimDecision(BaseModel):
    decision: str = Field(description="The final decision, either 'Approved' or 'Rejected'.")
    amount: str = Field(description="The approved payout amount, or 'N/A' if not applicable.")
    justification: str = Field(description="Detailed explanation citing specific clause numbers.")
    clauses_referenced: List[str] = Field(description="Verbatim text of the policy clauses used for the decision.")

class QueryRequest(BaseModel):
    query: str
    policy_name: str  

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    temperature=0,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

parser = JsonOutputParser(pydantic_object=ClaimDecision)

prompt_template_str = """
You are an expert insurance claims processing agent. 
Evaluate the query based ONLY on the provided policy clauses.

CONTEXT (Clauses from {policy_name}):
{context}

QUERY:
{query}

INSTRUCTIONS:
1. Analyze the query against the context.
2. If the context does not contain enough info, state that in the justification.
3. Reference specific clause numbers in your justification.
4. Output strict JSON.

{format_instructions}
"""

prompt_template = PromptTemplate(
    template=prompt_template_str,
    input_variables=["context", "query", "policy_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

def format_docs_with_sources(docs: List[Document]) -> str:
    """Formats retrieved docs with Source and Page info for the LLM."""
    formatted = []
    for i, doc in enumerate(docs):
        page = doc.metadata.get('page', 'N/A')
        source = doc.metadata.get('source', 'Unknown')
        content = doc.page_content.replace("\n", " ")
        formatted.append(f"--- Clause {i+1} (Page {page}) ---\n{content}")
    return "\n\n".join(formatted)

def get_available_policies():
    """Returns a list of available index names."""
    if not os.path.exists(INDEX_PATH_BASE):
        return []
    # List subdirectories in vector_store
    return [d for d in os.listdir(INDEX_PATH_BASE) 
            if os.path.isdir(os.path.join(INDEX_PATH_BASE, d))]

app = FastAPI(title="Insurance RAG System")

@app.post("/process_claim", response_model=ClaimDecision)
async def process_claim(request: QueryRequest):
    """
    Process a claim against a specific policy document.
    """
    target_index_path = os.path.join(INDEX_PATH_BASE, request.policy_name)
    
    if not os.path.exists(target_index_path):
        available = get_available_policies()
        raise HTTPException(
            status_code=404, 
            detail=f"Policy '{request.policy_name}' not found. Available policies: {available}"
        )

    try:
        db = FAISS.load_local(
            target_index_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={'k': 4})

        chain = (
            {
                "context": retriever | format_docs_with_sources, 
                "query": RunnablePassthrough(),
                "policy_name": lambda x: request.policy_name
            }
            | prompt_template
            | llm
            | parser
        )

        result = chain.invoke(request.query)
        return result

    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/policies")
async def list_policies():
    """Helper endpoint to see what documents are indexed."""
    return {"available_policies": get_available_policies()}