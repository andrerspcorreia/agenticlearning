from base64 import b64encode

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

# Load environment variables from .env file
load_dotenv()

# Create the model reference by providing its identifier
model = init_chat_model(
    model = 'gpt-4.1-mini', # mistral-medium-2508
    temperature = 0.1
)

messages = [{
    "role" : "user",
    "content" : [
        {
            "type" : "text",
            "text" : "Describe the contents of this image."
        },
        {
            "type" : "image",
            "url" : "https://andrerspcorreia.github.io/andrerspcorreia/images/my_photo.jpg"
        }
    ]
}]

response = model.invoke(messages)
print(response.content)

messages = [{
    "role" : "user",
    "content" : [
        {
            "type" : "text",
            "text" : "Describe the contents of this image."
        },
        {
            "type" : "image",
            "base64" : b64encode(open("my_photo.jp", "rb").read()).decode(), # Encode the image bytes as b64 and then decode into a string
            "mime_type" : "image/jpg"
        }
    ]
}]

response = model.invoke(messages)
print(response.content)