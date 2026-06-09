import time

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables from .env file
load_dotenv()

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

# Initialize an agent, give it a starting model it doesn't matter which, and the middleware
agent = create_agent(
    model = 'gpt-4.1-mini',
    middleware = [HooksDemo()]
)     

# Call the agent
response = agent.invoke(
        {
        "messages" : [
            SystemMessage("You are a helpful assistant."),
            HumanMessage("What is 1 + 1?")
        ]
    }
)

print(response["messages"][-1].content)
print(response["messages"][-1].response_metadata["model_name"])