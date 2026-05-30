# Agentic Learning
My self-taught journey on Agentic Systems, LangChain and LangGraph.

## 1 - LangChain Overview

Following this [video tutorial](https://www.youtube.com/watch?v=J7j5tCB_y4w) which provides a general overview.

### 1.0 Setup Environment
Setup the environment and install the packages of the provider you are going to use. 
```
uv init # Initialize a virtual environment
uv add langchain # Install package
uv add "langchain[openai]" # Install package with OpenAI dependencies
uv add "langchain[anthropic]" # Same for Anthropic
uv add "langchain[mistralai]" # Same for MistralAI
uv add "langchain[google-genai]" # Same for Google
```
Create a file with API key you are going to use.
```
OPENAI_API_KEY=
MISTRAL_API_KEY=
ANTROPIC_API_KEY=
GOOGLE_API_KEY=# may not be the correct name
```
Install this package to load the variables from the **.env** file.
```
uv add python-dotenv
```

### 1.1 Basic Agent with a tool

1. Load **.env** API Keys.
```
load_dotenv()
```
2. Declare a function as a **tool**.
```
@tool("get_weather", description = "Get the current weather for a city")
def get_weather(city: str) -> dict:
```
3. Initialize an **Agent**: 
    1. Specify the LLM name - automatically uses the API Key.
    2. List of tools - list of functions declared as tools.
    3. System prompt - a string telling the LLM its role.
```
agent = create_agent(
    model = "gpt-4.1-mini", 
    tools = [get_weather],
    system_prompt = "You are a weather assistant." 
)
```
4. Create a dictionary variable for the **messages** with the following structure: 
```
messages = {
    "messages" : 
    [
        {
            "role" : "user", 
            "content" : "<Prompt 1 here>"
        },
        {
            "role" : "user", 
            "content" : "<Prompt 2 here>"
        },
        ...
    ]
}
```
5. Call **invoke()** to obtain the LLM's response to each prompt, using the available tools.
```
response = agent.invoke(messages)
# Content of the last message
response["messages"][-1]["content"]
```