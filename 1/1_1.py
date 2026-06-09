import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool


# Load environment variables from .env file
load_dotenv()

@tool("get_weather", description = "Get the current weather for a city", return_direct = False)
def get_weather(city: str) -> dict:
    # Obtain a JSON object with weather information for the specified city
    response = requests.get(f"https://wttr.in/{city}?format=j1")

    return response.json()

agent = create_agent(
    model = "gpt-4.1-mini", # Imediately recognized as OpenAI, requires OPENAI_API_KEY in .env and langchain[openai] package
    tools = [get_weather],  # List of tools the agent can use
    system_prompt = "You are a helpful weather assistant who always provides cracks jokes and is humourous while remaining helpful." 
)
response = agent.invoke({
    "messages" : [
        {
            "role": "user", 
            "content": "What's the weather like in Vienna?"
        }
    ]
})
print(response)
print(response["messages"][-1]["content"])