import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.messages import HumanMessage, AIMessage, SystemMessage


# Load environment variables from .env file
load_dotenv()

# Create the model reference by providing its identifier
model = init_chat_model(
    model = 'gpt-4.1-mini', # mistral-medium-2508
    temperature = 0.1
)

# A conversation is a list of messages: System (agent prompting), Human or AI (LLM output)
conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming."),
    HumanMessage("What is python?"),
    AIMessage("Python is an interpreted programming language."),
    HumanMessage("When was it released?")
]

# Prompt the model and obtain a response object
response = model.invoke(conversation)

# Show the entire response object
print(response)

# Show solely the text of the response
print(response.content)