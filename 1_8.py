from dataclasses import dataclass

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, ModelResponse, dynamic_prompt

# Load environment variables from .env file
load_dotenv()

@dataclass
class Context:
    user_role : str

@dynamic_prompt
def user_role_prompt(request : ModelRequest) -> str:
    user_role = request.runtime.context.user_role

    base_prompt = "You are a helpful and very concise assistant."

    match user_role:
        case "expert":
            return f"{base_prompt} Provide technical detail responses."
        case "beginner":
            return f"{base_prompt} Keep your explanations simple basic."
        case "child":
            return f"{base_prompt} Explain everything as if you were literally talking to a five year old."
        case _:
            return base_prompt

agent = create_agent(
    model = "gpt-4.1-mini", # Imediately recognized as OpenAI, requires OPENAI_API_KEY in .env and langchain[openai] package
    middleware = [user_role_prompt],
    context_schema = Context
)         

response = agent.invoke(
        {
        "messages" : [
            {
                "role": "user", 
                "content": "Explain PCA"
            }
        ]
    },
    context = Context(user_role = "child")
)
print(response)
print(response["messages"][-1]["content"])