from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

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

# Create the LLM access object
llm = ChatGroq(model = "llama-3.3-70b-versatile", temperature = 0)

agent = create_agent(
    llm,
    tools = [get_product],
    system_prompt = "You are a helpful product assistant for an online tech store.",
    checkpointer = InMemorySaver()
)

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

ask("what is the price of wireless headphones.")
ask("what are the reviews on this product")