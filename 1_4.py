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

# Prompt the model and obtain a response as it's being generated (chunk by chunk)
for chunk in model.stream("Hello, what is Python?"):
    print(chunk.text, end  ="", flush = True) # end = "" to not have line breaks after each print, flush = True to see the output in realtime