from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent
import base64

load_dotenv()

# Open the image as a string
with open("blood_work.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()


# Create the LLM access object
llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct")

# Create the message to send to LLM
message = HumanMessage(
    content = 
    [
        {
            "type": "image_url", 
            "image_url": {
                "url": f"data:image/png;base64,{image_b64}"
            }
        },
        {
            "type": "text",      
            "text": "This is a blood work report. Extract all test results and flag any values outside the normal range."
        }
])

# Invoke the model with the image and text
response = llm.invoke([message])
print(response.content)