import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


# Load environment variables from .env file
load_dotenv()

# Create the model reference by providing its identifier
model = init_chat_model(
    model = 'gpt-4.1-mini', # mistral-medium-2508
    temperature = 0.1
)

# Prompt the model and obtain a response object
response = model.invoke("Hello, what is Python?")

# Show the entire response object
print(response)

# Show solely the text of the response
print(response.content)