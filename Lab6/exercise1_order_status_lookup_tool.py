from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b",api_key=os.environ["GROQ_API_KEY"])


@tool
def get_order_status(order_id:str)->str:
    """Get the shipping status of an order"""
    orders={
        "ORD-4521": "Shipped and in transit",
        "ORD-1234": "Delivered",
        "ORD-7890": "Processing"
    }
    return orders.get(order_id,"order not found")

llm_with_tool = llm.bind_tools([get_order_status])
user_query = input("Enter question: ")
response = llm_with_tool.invoke(user_query)
print("\nTool call: ",response.tool_calls)

if response.tool_calls:
    tool_call = response.tool_calls[0]
    result=get_order_status.invoke(tool_call["args"])
    print("\nTool result: ",result)

    final_res = llm.invoke(f"User Question: {user_query}\nTool Result: {result}\nGive a simple natural language answer")
    print("Final Answer: ",final_res.content)