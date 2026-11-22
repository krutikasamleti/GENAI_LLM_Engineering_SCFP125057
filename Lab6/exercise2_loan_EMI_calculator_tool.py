from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="openai/gpt-oss-20b",api_key=os.environ["GROQ_API_KEY"])

#pydantic model for structured tool
class EMIInput(BaseModel):
    principal:float = Field(gt=0, description="The principal loan amount")
    annual_rate:float = Field(ge=0,le=30, description="The annual interest rate in percentage")
    tenure_months:int = Field(gt=0, description="The loan tenure in months")

def calculate_emi(principal,annual_rate,tenure_months):
    """Calculate the EMI for a loan"""
    if annual_rate == 0:
        return round(principal / tenure_months,2)

    monthly_rate = annual_rate / 12 / 100

    emi = (principal*monthly_rate*(1+monthly_rate)**tenure_months)/((1+monthly_rate)**tenure_months-1)
    return round(emi,2)

emi_tool = StructuredTool.from_function(
    func = calculate_emi,
    name = "calculate_emi",
    description = "Calculate the EMI for a loan",
    arg_model = EMIInput
)

llm_with_tool = llm.bind_tools([emi_tool])
user_query = input("Enter question: ")
response = llm_with_tool.invoke(user_query)
print("\nTool call: ",response.tool_calls)

#Manually execute tool result
if response.tool_calls:
    tool_call = response.tool_calls[0]
    result = emi_tool.invoke(tool_call["args"])
    print("\nTool result: ",result)

    final_res = llm.invoke(
        f"""
        User Question: {user_query}
        EMI Calculated by the tool: {result}
        Give a simple natural-language answer to the user.
        """
    )
    print("\nFinal Answer: ",final_res.content)