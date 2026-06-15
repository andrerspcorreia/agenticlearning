from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent

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

# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

agent = create_agent(
    llm,
    tools=[get_product],
    system_prompt="You are a helpful product assistant for an online tech store.",
)

def ask(question: str):
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    print(result["messages"][-1].content)

ask("what is the price of wireless headphones.")