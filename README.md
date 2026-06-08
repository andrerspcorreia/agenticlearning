# Agentic Learning

My self-taught journey on Agentic Systems, LangChain and LangGraph.

## 1 - LangChain Overview

Following this [video tutorial](https://www.youtube.com/watch?v=J7j5tCB_y4w) which provides a general overview of LangChain.

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

### 1.2 Direct LLM inference (no agent)

How to interact with an LLM (like chat-GPT) but through code:

1. Load **.env** API Keys.

```
load_dotenv()
```

2. Create the **model reference** using its identifier

```
model = init_chat_model(
    model = 'MODEL NAME HERE',
    temperature = 0.1
)
```

3. **Prompt** the model and obtain a response object

```
response = model.invoke("PROMPT MESSAGE HERE")
response.content
```

### 1.3 LLM conversation (sequence of messages)

How to have **maintain a memory**?

How to have a **conversation** with more than one message?

1. Load **.env** API Keys.

```
load_dotenv()
```

2. Create the **model reference** using its identifier

```
model = init_chat_model(
    model = 'MODEL NAME HERE',
    temperature = 0.1
)
```

3. Create and maintain a **conversation** object.

A **conversation** is a list of messages: **System** (for example to prompt the agent), **Human** (what to ask the AI) or **AI** (responses from the AI)

```
conversation = [
    SystemMessage("You are a helpful assistant for questions regarding programming."),
    HumanMessage("What is python?"),
    AIMessage("Python is an interpreted programming language."),
    HumanMessage("When was it released?")
]
```

4 Prompt the model on the conversation

```
response = model.invoke(conversation)
response.content

```

### 1.4 Real Time Output (Stream output)

How to view the text **as it is being generated** by the model.

**Instead of waiting** for all the text to be generated, and then showing it.

1. Prompt the model and obtain a response as it's **being generated**.

```
for chunk in model.stream("PROMPT MESSAGE HERE"):
    print(chunk.text, end  ="", flush = True)
```

### 1.5 Chat model AND agent, Context, ResponseFormat, Checkpointer

You can initialize your LLM model like in **1.3**:

```
model = init_chat_model("gpt-4.1-mini", temperature = 0.1)
```

Then, use this model to create the agent, instead of passing the LLM's name to the agent and losing access to the LLM.

```
agent = create_agent(model = model)
```

You can define a class to force the agent to structure its response in a specific format.

```
@dataclass
class ResponseFormat:
    summary : str
    temperature_celsius : float
    temperature_fahrenheit : float
    humidity : float
```

You define the format at the moment of initializing the agent.

```
agent = create_agent(
    model = model,
    response_format = ResponseFormat # Your class
)
```

You can define a dataclass that holds information about the context, for example a userID.

```
@dataclass
class Context:
    user_id : str
```

Similarly, define the context class at the moment of initializing the agent.

```
agent = create_agent(
    model = model,
    context_schema = Context # Your class
)
```

This class can be used to pass context to tool functions like so.

```
@tool("example_tool", description = "What the model should do with this tool")
def example_tool(runtime : ToolRuntime[Context]):
    # Do something with the context object like runtime.user_id
```

A checkpointer allows you to easily handle separate conversations with the agent using their own thread IDs.

```
checkpointer = InMemorySaver()
agent = create_agent(
    model = model,
    checkpointer = checkpointer
)
```

We can build the agent with all the information we've talked about like so:

```
agent = create_agent(
    model = model,
    tools = [get_weather, locate_user],  # List of tools the agent can use
    system_prompt = "You are a helpful weather assistant who always provides cracks jokes and is humourous while remaining helpful.",
    context_schema = Context, # here is where we define the dataclass of our context
    response_format = ResponseFormat, # same logic here for the response format, here we define its class
    checkpointer = checkpointer
)
```

This agent has access to:

- The LLM object.
- Two tools, defined by their descriptions.
- A system prompt: condition the agent on its role.
- Context class: provides the ability to pass contexts to the agent.
- A response format: to force the agent to answer in a specific format
- Checkpointer: memory so that the agent can keep track of multiple conversations based on a thread ID.

Before calling the model create a configuration dictionary with the threadID (similarly for context).

```
config = {
    "configurable": {
        "thread_id" : 1 # Thread ID here
    }
}
context = Context(
    user_id = "USER_ID" # Context variables here
)

response = agent.invoke(
    {
        "messages" : [
            {
                "role": "user",
                "content": "YOUR MESSAGE PROMPT HERE"
            }
        ]
    },
    config = config,
    context = context
)
```

Show the response (dataclass object), or a variable of it.

```
print(response["structured_response"])
print(response["structured_response"].temperature_celsius)
```

### 1.6 Multi-modal input: Text AND images

How to pass an image from an URL

```
from base64 import b64encode
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
```

How to pass an image from disk. We need to load the image bytes, encode the image bytes as b64, and then decode them into a string to pass to the model.

```
from base64 import b64encode
messages = [{
    "role" : "user",
    "content" : [
        {
            "type" : "text",
            "text" : "Describe the contents of this image."
        },
        {
            "type" : "image",
            "base64" : b64encode(open("my_photo.jp", "rb").read()).decode(),
            "mime_type" : "image/jpg"
        }
    ]
}]
response = model.invoke(messages)
print(response.content)
```
