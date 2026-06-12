from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

PDF_PATH = "telecom_guide.pdf"

load_dotenv()

# Load the PDF
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
print(f"Loaded {len(pages)} pages from the PDF.")

# Split the PDF into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600, # words per chunk
    chunk_overlap = 100, # overlap 100 words between chunks
    separators = ["\n\n", "\n", ".", " "] # tries paragraph -> sentence -> word
)
chunks = splitter.split_documents(pages)
print(len(chunks))

# Convert them to Embeddings
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)

# Create the retriever
retriever = vector_store.as_retriever(search_kwargs = {"k" : 3}) # Return the TOP 3 chunks

# Query the retriever
test_query = "What is VOLTE and how does it improve call quality?"
retrieved = retriever.invoke(test_query)

# RAG Pipeline
# System Prompt
SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])

# Create the LLM access object
llm = ChatGroq(
    model = "qwen/qwen3-32b",
    temperature = 0,
    reasoning_format = "parsed",
)

# Assemble the chain
# --- Helper: join retrieved chunks into a single context string ---
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("RAG chain assembled.")

# Invoke the Chain
question = "How does international roaming work and what charges should I expect?"
answer = chain.invoke(question)
print(answer)