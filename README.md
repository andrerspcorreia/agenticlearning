# Agentic Learning

My self-taught journey on Agentic Systems, [LangChain](https://www.docs.langchain.com) and LangGraph.

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

Create a file named **.env** with API key you are going to use.

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

### 2.1 - API Keys and Environment Setup

The course asks us to create both [Google](https://aistudio.google.com/) and [Grok](https://www.console.groq.com) API Keys. Create a **.env** file with them inside.

```
GOOGLE_API_KEY=
GROQ_API_KEY=
```

Create a **pyproject.toml** file inside the directory with the following contents.

```
[project]
name = "agentic-ai-crash-course"
version = "0.1.0"
description = "Agentic AI crash course by Codebasics"
readme = "README.md"
requires-python = ">=3.13"
dependencies = [
    "langchain>=0.3.0",
    "langchain-core>=1.0.0",
    "langchain-google-genai>=4.2.2",
    "langchain-groq>=0.3.0",
    "python-dotenv>=1.2.2",
    "streamlit>=1.57.0",
    "ipykernel>=7.2.0",
    "notebook>=7.5.6",
    "langchain-chroma>=1.1.0",
    "langchain-huggingface>=1.2.2",
    "chromadb>=1.5.9",
    "sentence-transformers>=5.5.0",
    "pandas>=3.0.2",
    "fpdf2>=2.8.7",
    "pypdf>=6.11.0",
    "langchain-community>=0.4.1",
    "langchain-text-splitters>=1.1.2",
    "pillow>=12.2.0",
    "torchvision>=0.27.0",
]
```

Run the environment installation commands.

```
uv init
uv sync
```

### 2.2 - Simple LLM Calls

**Google** Gemini example.

```
llm = ChatGoogleGenerativeAI(model = "gemma-4-31b-it", temperature = 0)
response = llm.invoke("How many moons does Jupiter have?")
print(response.text)
```

**System message** example.

```
response = llm.invoke([
    ["system", "You are a helpful assistant that answers in one line."],
    ["human", "How many moons does Jupiter have?"]
])
```

**Groq** QWEN3 call example.

```
llm = ChatGroq(model = "qwen/qwen3-32b", temperature = 0)
```

### 2.3 List Gemini Models

```
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

genai.configure(api_key = os.environment["GOOGLE_API_KEY"])
for model in genai.list_models():
    print(model.name)
```

### 2.4 Workflow Example

Example of parsing some information from a source (text file for example), passing it to an LLM, then the result goes also back to an LLM to obtain the final result.

Read the file for the information.

```
with open("blood_work.txt", "r") as f:
    blood_report = f.read()
```

Create the LLM access object.

```
llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")
```

Prompt the LLM using the text information and obtain the first response.

```
extraction_prompt = f"""
You are a medical data extraction assistant.

From the blood report below, extract ALL test values and classify each one as HIGH, LOW, or NORMAL
based on the reference ranges provided in the report.

Format your response as:
- Test Name: value | Status: HIGH/LOW/NORMAL | Reference: range

Blood Report:
{blood_report}
"""

extraction_response = llm.invoke(extraction_prompt)
extracted_values = extraction_response.text
```

Create a second prompt with the previous LLM response to obtain the final results.

```
diet_prompt = f"""
You are a clinical nutritionist specializing in Indian dietary habits.

Based on the blood work analysis below, write:
1. A short health summary in 4-5 lines explaining the patient's condition in simple language
2. A short, practical Indian diet plan having only two sections (1) Foods to avoid (2) Foods to eat more of.
   Do not include any other sections in diet plan.

Blood Work Analysis:
{extracted_values}
"""
diet_response = llm.invoke(diet_prompt)
```

### 2.5 RAG - Retrieval Augmented Generation

LLMs are trained on general knowledge. It does not have access to private information and may struggle with specific information. But if you provide it some source of information, it can infer the answer.

However, the LLM is limited to its context window so you may not pass it all of your information. Additionally, this would be very costly over time.

Instead you can pass it chunks of the information which are likely to contain the answer.
How do you know what chunks to pass it? Embeddings. Convert the text into a vector such that it can represent their meaning in number form.

To generate embeddings you can use a variety of models such as sentence-transformers or text-embedding-3-small. You can store these embeddings in a vector based database such as Mivus, Qdrant, ChromaDB. This step is called indexing, where you index these chunks into a vector database.

The second step is retrieval, where for a given question you generate its embedding using the same embedding model and then you try to find the relevant chunks in the database.

Then you obtain the text of those selected chunks, put them inside a prompt to query the LLM and obtain the final answer.

Benefits of RAG:

- Higher Accuracy answers.
- Fewer Hallucinations.
- Cost Effective (less tokens sent).

#### 2.5.1 Vector Databases (ChromaDB)

The popular opensource vector database is ChromaDB.

```
uv add chromadb
```

Import the library and create a Client which works like the Database.

```
import chromadb

client = chromadb.Client()
```

Now create a collection which will contain information and works like a Table.

```
collection = client.create_collection("news")
```

Add the information to the collection as Documents, these work like rows.

ChromaDB will create embeddings for each document and store them.

By default ChromaDB uses the all-MiniLM-L6-v2 embedding model

```
collection.add(
    ids = ["id1", "id2", "id3", "id4"],
    documents=[
        "Apple is leading in a smart phone game with iPhone sales up by 35%",
        "Tesla booked a minor profit of 1 billion $ in Q2",
        "Apples are high in fiber, vitamin C, and various antioxidants",
        "SpaceX got NASA contract worth 10 billion $",
    ]
)
```

You obtain these embeddings from the collection and see how they look.

```
data = collection.get(include = ["documents", "embeddings", "metadatas"])
for emb in data["embeddings"]:
    print(emb)
```

Finally, you can query the database and obtain the closest matched documents it contains. It does semantic search. In other words it searches through the meaning of the document instead of keywords or words. In these examples, we don't have any document with words Elon or Musk. Yet, it will know to fetch the documents related to Tesla and SpaceX.

```
results = collection.query(query_texts = ["This is a query related to Elon Musk."], n_results = 2)
print(results)
```

#### 2.5.2 Pipeline Example

Load information for example from a PDF.

```
loader = PyPDFLoader(PDF_PATH)
pages = loader.load()
```

Split the information into chunks. Specify the size of each chunk (number of words). Number of overlapping words between consecutive chunks. Separators that define what constitutes a word.

```
splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600, # words per chunk
    chunk_overlap = 100, # overlap 100 words between chunks
    separators = ["\n\n", "\n", ".", " "] # tries paragraph -> sentence -> word
)
chunks = splitter.split_documents(pages)
```

Convert the chunks into embeddings, and convert the embeddings into a vector store (database).

```
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
vector_store = Chroma.from_documents(chunks, embeddings)
```

Create the retriever using the vector store (database)

```
retriever = vector_store.as_retriever(search_kwargs = {"k" : 3}) # Return the TOP 3 chunks
```

You can query the retriever, similar to querying the database before.

```
retrieved = retriever.invoke("What is VOLTE and how does it improve call quality?")
```

Let's create the LLM pipeline. First, create the LLM access object.

```
llm = ChatGroq(
    model = "qwen/qwen3-32b",
    temperature = 0,
    reasoning_format = "parsed",
)
```

Now lets define the first prompt.

```
SYSTEM_PROMPT = """\
You are a helpful telecom assistant.
Answer the question using ONLY the context provided below.
If the context does not contain enough information, say so clearly.

Context:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])
```

Create the chain (pipeline). Here the function is a helpful parser.

```
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Finally, invoke the chain.

```
question = "How does international roaming work and what charges should I expect?"
answer = chain.invoke(question)
```

### 2.6 Agents

Agents = LLM + Tools + Memory

#### 2.6.1 Tools

Here's an example of a few tools. **It is the document string in the function that allows the agent to figure out which tool to use!**

```
PRODUCTS = {
    "wireless headphones": {"price": 79.99,  "rating": 4.6, "description": "Over-ear Bluetooth, 30-hr battery, active noise cancellation."},
    "smart watch":         {"price": 199.99, "rating": 4.3, "description": "Tracks heart rate and sleep. 5-day battery, water-resistant."},
    "mechanical keyboard": {"price": 129.00, "rating": 4.8, "description": "Tenkeyless, Cherry MX Brown switches, per-key RGB."},
    "laptop stand":        {"price": 34.99,  "rating": 4.5, "description": "Adjustable aluminium, fits 11-17 inch laptops, folds flat."},
}

@tool
def get_product(name: str) -> str:
    """Look up a product by name and return its price, rating, stock, and description."""
    p = PRODUCTS.get(name.lower())
    if not p:
        return f"Product not found. Available: {', '.join(PRODUCTS)}"
    return str(p)

REVIEWS = {
    "wireless headphones": {"reviews": 1262, "rating": 4.6},
    "smart watch":         {"reviews": 340,  "rating": 3.9},
    "mechanical keyboard": {"reviews": 67,   "rating": 4.8},
    "laptop stand":        {"reviews": 781,  "rating": 4.5},
}

@tool
def get_review(name: str) -> str:
    """Look up a product review by a product name. Return the product name, number of reviews and rating"""
    r = REVIEWS.get(name.lower())
    if not r:
        return f"Review not available for this product"
    return str(r)
```

Create the agent in the way that we already know. In this case we choose to have access to both the LLM and the agent.

```
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 0)

agent = create_agent(
    llm,
    tools = [get_product],
    system_prompt = "You are a helpful product assistant for an online tech store.",
)
```

Let's place an agent call inside a function for easy use.

```
def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)
```

Now, we can easily use the agent just with a prompt string.

```
ask("what is the price of wireless headphones.")
```

The agent uses its LLM to look at the question and its tool's descriptions (doc strings), and figures out which tool to use. It knows to extract the product name and call the right tool.

1 - Extract product name.

2 - Call **get_product("wireless headphones")** tool.

3 - Analyze the answer of the tool.

4 - Return human readable answer.

#### 2.6.2 Memory

Here's how you give an "in" memory to your agent.

"in" memory - session memory it does not persist to disk.

```
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    llm,
    tools = [get_product],
    system_prompt = "You are a helpful product assistant for an online tech store.",
    checkpointer = InMemorySaver()
)
```

Here's how you call the agent with a memory. Just like the first course, we give the agent a thread ID.

```
def ask(question: str):
    config = {
        "configurable":
        {
            "thread_id":
            "user-alice-session-1"
        }
    }
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": question
                }
            ],
        }, config
    )
    print(result["messages"][-1].content)
```

Now we can ask the agent two questions, and the agent can answer the second one based on the first!

```
ask("what is the price of wireless headphones.")
ask("what are the reviews on this product")
```

#### 2.6.3 Multi-Agent Systems

When there is a task too great for a single agent you may employ multiple. For example coding agents: an agent to plan, another to write the code, another to review and another to test.

#### 2.6.4 Multi-Modal Agents

AI model that can process more than one type of data, for example text and images. You open the image bytes and convert them to string. Then you pass the image string and a text prompt to the model.

```
with open("blood_work.png", "rb") as f:
    image_b64 = base64.b64encode(f.read()).decode()

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
```

#### 2.6.5 Evaluation

AI Agents are probabilistic - they can produce different output for the same input.

- Functional Evaluation - Is the answer correct? Hallucinations?
- Cost Evaluation - Number of tokens, latency.
- Safety Evaluation - Toxic output, jailbreaks.

Langsmith and Ragas are two frameworks you can use for Evaluation.

Convert both the LLM answer and desired answer to embeddings. Then do cosine similarity between them.

Create an account in [smith.langchain.com](https://www.smith.langchain.com).
Then create an API key and place it in the .env file as "LANGSMITH_API_KEY".

#### 2.6.6 Guard Rails

Your agent may have to be restricted to not access or leak critical information. Or out of scope or jailbreaks. Here's how to add guard rails using Middelware.

```
agent = create_agent(
    system_prompt="""You are a customer service assistant.
    You have access to the get_customer_info_tool which provides customer information based on the customer's name.
    When a user asks for information about a customer, use the get_customer_info_tool to retrieve the data.

    IMPORTANT RULES:
    1. Return all data fields EXACTLY as received from the tool — do NOT reformat,
       abbreviate, paraphrase, or restructure any field values (especially numbers and emails).
    2. Credit card numbers must always be returned in their original format with dashes
       e.g., XXXX-XXXX-XXXX-XXXX — never remove dashes or spaces.
    3. Do not handle PII directly. Middleware will automatically redact/mask sensitive
       fields — your job is only to pass the raw values through unchanged.
    4. Return information as plain text, not JSON.
    """,
    model = llm,
    tools = [get_customer_info_tool],
    middleware= [       # Mask credit cards in user input
        PIIMiddleware(
            "credit_card",
            strategy = "mask",
            apply_to_tool_results = True
        ),
        # Redact emails in user input before sending to model
        PIIMiddleware(
            "email",
            strategy = "redact",
            apply_to_tool_results = True,
            apply_to_output = True
        ),
    ]
)
```

## Cheat Sheet

1 - **Install packages** - install langchain and then [specific llm suplier]

2 - **Get keys and create file** - put API keys in .env file

3 - **load_dotenv** - always call this at the beginning it will load api keys and you don't have to worry about them anymore

4 - **Create an agent with an LLM** - use the create_agent function and specify llm name

5 - **Create a tool and an agent with a tool** - create a function with annotator tool and the docstring tells the agent what it does, the tool is passed as a list to the agent in the create_agent function

6 - **Create an llm** - init_chat_model creates an llm connection can be passed to create_agent instead of the string name

7 - **Create messages and call the agent** - the dictionary way with "messages" pointing to a list, or a list system/human/aimessage objects (easier)

8 - **Create more than 1 message** - with the second method described above

9 - **Realtime output (stream)** - use this: for chunk in model.stream("PROMPT MESSAGE HERE"):

10 - **Context** - create a dataclass, pass it to the create_agent, this should now be the input of a tool

11 - **Response format** - create a dataclass, pass it to the create_agent, agent should now respond in this format

12 - **Checkpointer** - create the object InMemorySaver(), pass it to the agent, call it with a config object with thread id key

13 - **Multimodal input** - can pass images with url and bytes, don't memorize just copy paste from example how to do it

14 - **RAG** - source of knowledge -> convert to vector/embedding -> create a retriever from this -> have model access this retriever (for example as a tool or inside a tool)

15 - **Middleware** - pass them inside create_agent as a list similar to tools, existing middleware or custom, custom has many annotators that happen at different points of the pipeline
