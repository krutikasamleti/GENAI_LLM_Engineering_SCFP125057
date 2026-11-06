from langchain_groq import ChatGroq
from langchain_core.tools import BaseTool
from dotenv import load_dotenv
import os
import asyncio

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b",api_key=os.environ["GROQ_API_KEY"])



