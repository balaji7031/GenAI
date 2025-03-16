from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo


import os
from dotenv import load_dotenv

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

#websearch agent

web_search_agent=Agent(
    name='web_search_agent',
    role="search the web for information",
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],
    show_tool_calls=True,
    markdown=True
    )

#financial agent

financial_agent=Agent(
    name='financial_agent',
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True)],
    show_tool_calls=True,
    description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
    instructions=["Format your response using markdown and use tables to display data where possible."],
    markdown=True
    )

#multi ai agent

multi_ai_agent=Agent(
    team=[web_search_agent, financial_agent],
    model=Groq(id="llama-3.3-70b-versatile",api_key=GROQ_API_KEY),
    instructions=["always include sources,use tables to display the data"],
    markdown=True,
    show_tool_calls=True
    )


multi_ai_agent.print_response("summarize analyst recommendations and share the latest news foor NVDA",stream=True)

