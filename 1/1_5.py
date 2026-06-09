from dataclasses import dataclass

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

# Load environment variables from .env file
load_dotenv()

@dataclass
class Context:
    user_id : str

@dataclass
class ResponseFormat:
    summary : str
    temperature_celsius : float
    temperature_fahrenheit : float
    humidity : float

@tool("get_weather", description = "Get the current weather for a city", return_direct = False)
def get_weather(city: str) -> dict:
    # Obtain a JSON object with weather information for the specified city
    response = requests.get(f"https://wttr.in/{city}?format=j1")

    return response.json()

@tool("locate_user", description = "Look up a user's city based on the context.")
def locate_user(runtime : ToolRuntime[Context]):
    # runtime of object ToolRuntime is passed to this tool which is based on the data class context which we defined above
    match runtime.context.user_id:
        case "ABC123":
            return "Vienna"
        case "XYZ456":
            return "London"
        case _:
            return "Unknown" # You can provide unknown behavior in multiple ways. In the description, in the return value, in the system prompt and custom logic.

model = init_chat_model("gpt-4.1-mini", temperature = 0.1)
checkpointer = InMemorySaver() # to remember conversations

agent = create_agent(
    model = model, 
    tools = [get_weather, locate_user],  # List of tools the agent can use
    system_prompt = "You are a helpful weather assistant who always provides cracks jokes and is humourous while remaining helpful.",
    context_schema = Context, # here is where we define the dataclass of our context
    response_format = ResponseFormat, # same logic here for the response format, here we define its class
    checkpointer = checkpointer
)
# This agent has access to:
# - the LLM, 
# - two tools: one for getting weather information of a given city and another to locate a user.
# - a system prompt to give it context about its role
# - ability to pass contexts to the agent
# - response format to force the model to answer in a specific format
# - memory so it can keep track of conversations based on a thread ID

config = {
    "configurable": {
        "thread_id" : 1
    }
}
context = Context(
    user_id = "ABC123"
)

response = agent.invoke(
    {
        "messages" : [
            {
                "role": "user", 
                "content": "What's the weather like in Vienna?"
            }
        ]
    },
    config = config,
    context = context
)
print(response["structured_response"])
print(response["structured_response"].temperature_celsius)