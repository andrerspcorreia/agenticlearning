from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS # uv add langchain-community uv add faiss-cpu
from langchin_core.tools import create_retriever_tool

# Load environment variables from .env file
load_dotenv()

embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")

texts = [
    "I love apples.",
    "I enjoy oranges.",
    "I think pears taste very good."
    "I hate bananas.",
    "I dislike raspberries.",
    "I despise mangos.",
    "I love Linux.",
    "I hate Windows."
]

# Similarity search using FAISS
vector_store = FAISS.from_texts(texts, embeddings = embeddings)
print(vector_store.similarity_search("What fruits does the person like?", k = 3))
print(vector_store.similarity_search("What fruits does the person hate?", k = 3))

# Use the vector store as a retriever
retriever = vector_store.as_retriever(
    search_kwargs = {
        "k" : 3 # TOP 3 answers
    }
)

# Convert it to a tool you can pass to an agent
retriever_tool = create_retriever_tool(retriever, name = "kb_search", description = "Search the fruit knowledge base for information.")

# Create the agent with the tool
agent = create_agent(
    model = "gpt-4.1-mini", # Imediately recognized as OpenAI, requires OPENAI_API_KEY in .env and langchain[openai] package
    tools = [retriever_tool],  # List of tools the agent can use
    system_prompt = 
        "You are a helpful assistant. " \
        "For questions about Macs, apples, or laptops. " \
        "First call the kb_search tool to retrieve context, then answer succinctly. " \
        "Maybe you have to use it multiple times before answering."
)

# Query the agent with a prompt
response = agent.invoke({
    "messages" : [
        {
            "role": "user", 
            "content": "What three fruits does the person like and what three fruits does the person dislike?"
        }
    ]
})

# Show the result
print(response)
print(response["messages"][-1]["content"])