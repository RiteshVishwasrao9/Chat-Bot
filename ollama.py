# Chat Bot using olllama

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import ollama


import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# Langsmith Tracking

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2']="true"

# Creating Chat Bot

prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. Please response to the user queries"),
        ("user","Question:{question}")
    ]
)

# streamlit Framework

st.title("LangChain Chatbot Using Ollama API")
input_text=st.text_input("Search any topic you want")

# OpenAi LLM Call

llm=ollama(model="llama2")
output_parser=StrOutputParser()

#chain

chain=prompt|llm|output_parser
if input_text:
    st.write(chain.invoke({'question':input_text}))