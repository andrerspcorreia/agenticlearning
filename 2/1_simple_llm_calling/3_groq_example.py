from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model = "qwen/qwen3-32b", temperature = 0)
response = llm.invoke([
    ["system", "You are a helpful assistant that answers in one line."],
    ["human", "How many moons does Jupiter have?"]
])
print(response.text)