from IPython.display import Markdown, display
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import web_search, scrape
from dotenv import load_dotenv

load_dotenv()

#Model Initialization
llm = ChatGroq(model="qwen/qwen3.8-27b",temperature=0, max_tokens=500)

#Search Agent
def search_agent():
    return create_agent(model=llm, tools=[web_search], system_prompt="You are helpful search agent for research assistant")

#Scrape Agent
def scrape_agent():
    return create_agent(model=llm, tools=[scrape], system_prompt="You are helpful reader agent for research assistant")

#Writer_Chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert in writing research. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Methods used 
- Key Findings (List top 3 findings)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#Critic_Chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()