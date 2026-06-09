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

2. Create the **model reference** using its identifier, temperature = 0 for deterministic behavior.

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

Then, use this model to **create the agent**, instead of passing the LLM's name to the agent and losing access to the LLM.

```
agent = create_agent(model = model)
```

You can define a class to force the agent to **structure its response** in a specific **format**.

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

You can define a dataclass that holds information about the **context**, for example a userID.

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

This class can be used to **pass context** to tool functions like so.

```
@tool("example_tool", description = "What the model should do with this tool")
def example_tool(runtime : ToolRuntime[Context]):
    # Do something with the context object like runtime.user_id
```

A **checkpointer** allows you to easily handle **separate conversations** with the agent using their own **thread IDs**.

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

This agent has **access to**:

- The **LLM** object.
- **Two tools**, defined by their descriptions.
- A **system prompt**: condition the agent on its role.
- **Context class**: provides the ability to pass contexts to the agent.
- A **response format**: to force the agent to answer in a specific format
- **Checkpointer**: memory so that the agent can keep track of multiple conversations based on a thread ID.

Before calling the model create a configuration **dictionary** with the **threadID** (similarly for context).

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

How to pass an **image from an URL**.

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

How to pass an **image from disk**. We need to load the image bytes, encode the image bytes as b64, and then decode them into a string to pass to the model.

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

### 1.7 RAG - Retrieval Augmented Generation

The agent uses some form of **knowledge base** to answer the questions/prompts.

For example we can have a list of knowledge:

```
texts = [
    "I love apples.",
    "I enjoy oranges.",
    "I think pears taste very good."
    "I hate bananas.",
    "I dislike raspberries.",
    "I despise mangos.",
    "I love Linux.",
    "I hate Windows."
]
```

Convert the knowledge into an **vector representation**.

```
embeddings = OpenAIEmbeddings(model = "text-embedding-3-large")
vector_store = FAISS.from_texts(texts, embeddings = embeddings)
```

You can do **similarity search** using the vector store.

```
vector_store.similarity_search("What fruits does the person like?", k = 3)
vector_store.similarity_search("What fruits does the person hate?", k = 3)
```

You can **convert** this similarity search on the vector store (vector representation of your knowledge) into a **tool** that an agent can use.

```
retriever = vector_store.as_retriever(
    search_kwargs = {
        "k" : 3 # TOP 3 answers
    }
)
retriever_tool = create_retriever_tool(retriever, name = "kb_search", description = "Search the fruit knowledge base for information.")

agent = create_agent(
    model = "gpt-4.1-mini",
    tools = [retriever_tool],
    system_prompt =
        "You are a helpful assistant. For questions about Macs, apples, or laptops. First call the kb_search tool to retrieve context, then answer succinctly. Maybe you have to use it multiple times before answering."
)
```

The agent can use the tools **multiple times** per prompt for example we can ask it two questions that require it to use the tool twice.

```
response = agent.invoke({
    "messages" : [
        {
            "role": "user",
            "content": "What three fruits does the person like and what three fruits does the person dislike?"
        }
    ]
})
response["messages"][-1]["content"]
```

### 1.8 (1.9 & 1.10) Middleware

Sits **between** the prompt and the response. Allows you to enhance the capabilities of the agent. For example you can choose between different system prompts or models based on the context. You can summarize, rate limit etc.

We will need a context class once again. In this example the behavior of the agent will change based on the user's role instead of ID.

```
@dataclass
class Context:
    user_role : str
```

Similar to tools, we create a function that the agent calls. Lets start with **changing the system prompt**.

```
@dynamic_prompt
def user_role_prompt(request : ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    # Do whatever you need to the prompt based on the role
```

Here's how you **initialize the agent** with the function as middleware and how to invoke it with context.

```
agent = create_agent(
    model = "gpt-4.1-mini",
    middleware = [user_role_prompt],
    context_schema = Context
)

response = agent.invoke(
        {
        "messages" : [
            {
                "role": "user",
                "content": "YOUR PROMPT MESSAGE"
            }
        ]
    },
    context = Context(user_role = "YOUR USER ROLE CASE")
)
```

Here's an example on how to **dynamically select a model**.

Start by initializing multiple models.

```
basic_model = init_chat_model(
    model = 'gpt-4.0-mini',
    temperature = 0.1
)
advanced_model = init_chat_model(
    model = 'gpt-4.1-mini',
    temperature = 0.1
)
```

**Create the middleware** that handles model selection. This function happens when the agent is called.

```
@wrap_model_call
def dynamic_model_selection(request : ModelRequest, handler) -> ModelResponse:
    message_count = len(request.state["messages"])

    if message_count > 3:
        model = advanced_model
    else:
        model = basic_model

    request.model = model

    return handler(request)
```

**Create an agent** with that middleware. You can assign it any starting model, it will be changed dynamically when the agent is called.

```
agent = create_agent(
    model = basic_model,
    middleware = [dynamic_model_selection]
)
```

Now an example of **custom middleware**.
Customize what happens in each part:

- Before Agent
- Before Model
- After Model
- After Agent

Note that the middleware is passed as an **instance** and not the class.

```
class HooksDemo(AgentMiddleware):
    def __init__(self):
        super().__init__()
        self.start_time = 0.0

    def before_agent(self, state: AgentState, runtime):
        self.start_time = time.time()
        print("Before agent triggered")

    def before_model(self, state: AgentState, runtime):
        print("Before model")

    def after_model(self, state: AgentState, runtime):
        print("After model")

    def after_agent(self, state: AgentState, runtime):
        print(f"After agent triggered {time.time() - self.start_time}")

agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware = [HooksDemo()]
)
```

Now example of **existing middleware**, read the docs for available ones.

Example for **summarization**: summarize the conversation after 4000 tokens and keep the last 20 messages using an LLM.

```
agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware = [
        SummarizationMiddleware(
            model = 'gpt-4.1-mini',
            max_tokens_before_summary = 4000,
            messages_to_keep = 20,
            summary_prompt = "Summarize the most important key points of this conversation."
        )
    ]
)
```

## 2 - Agentic AI Course with LangChain

Following this [video tutorial](https://www.youtube.com/watch?v=D74el9mvNak) which is a 2h crash course on Agentic AI with LangChain.

**Context Window**: number of tokens the agent can receive.
**Temperature**: Randomness/Stochasticity (0 = deterministic behavior).
**Top K**: Sample from the top k classes (tokens) with highest probabilities.
**Top P**: Sample from the top P probabilities (summed probabilities up to P) of classes.
**LLM**: Trained on large corpus of data to learn the statistical probabilities.
**RLHF (Reinforcement Learning with Human Feedback)**: Method to train an LLM with RL where the feedback is given by the human. The LLM produces more than one answer and the human selects the better one. To avoid "toxicity", requires a lot of humans!
