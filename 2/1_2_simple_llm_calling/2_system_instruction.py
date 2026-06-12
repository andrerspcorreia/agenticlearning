from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemma-4-31b-it")
response = llm.invoke([
    ["system", "You are a helpful assistant that answers in one line."],
    ["human", "How many moons does Jupiter have?"]
])
print(response.text)