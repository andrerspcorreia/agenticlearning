from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(model = "gemma-4-31b-it", temperature = 0)
response = llm.invoke("How many moons does Jupiter have?")
print(response.text)