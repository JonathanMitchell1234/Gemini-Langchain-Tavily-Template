import os
import getpass
from langchain_community.retrievers import TavilySearchAPIRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAI, HarmBlockThreshold, HarmCategory
from dotenv import load_dotenv

# Set up API keys
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
load_dotenv()

os.environ["TAVILY_API_KEY"] = getpass.getpass(prompt="Enter TAVILY API Key: ")

# Set up the retriever
retriever = TavilySearchAPIRetriever(k=10)

# Fetch and format the context
def get_formatted_context(query):
    results = retriever.invoke(query)
    context_texts = [doc.page_content for doc in results]
    return " ".join(context_texts)

# Define your query (Modified for report-like structure)
query = "What are the best sports betting picks for games occuring tonight?"

# Fetch context using Tavily
context = get_formatted_context(query)

# Configure Google Generative AI (You already have this)
llm = GoogleGenerativeAI(
    model="gemini-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"), 
    safety_settings={
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

# Set up the refined prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are an AI sports betting analyst. Using the provided information, write a concise report choosing the best picks.

**Report Format**
* List each game with the teams playing.
* Include the start time for each game.
* Include the bets with the best chance of winning.
* Provide an extremely detailed summary with at least 500 words
* Always cite your sources

**Context:** {context}
"""
)

# Simplified chain - Remains the same
chain = (
    RunnablePassthrough()  
    | (lambda _: prompt_template.invoke({"context": context}))  # Removed 'question'
    | llm
    | StrOutputParser()
)

# Run the chain
result = chain.invoke({})
print(result) 
