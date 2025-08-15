from typing import TypedDict, Annotated, Optional, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import BaseMessage
import operator
import json

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from serpapi import GoogleSearch

from functools import partial
import os
from dotenv import load_dotenv

# Import your Pinecone query function
from features.pinecone_index import query_pinecone

load_dotenv()

# Environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

## Creating the Agent State ##
class AgentState(TypedDict):
    """State structure for the financial reports agent"""
    input: str
    chat_history: list[BaseMessage]
    intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
    year: Optional[str]
    quarter: Optional[List[str]]
    org: Optional[List[str]]  # Added org for filtering by company
    

@tool("vector_search")
def vector_search(query: str, year: str = None, quarter: list = None, org: list = None):
    """
    Searches for the most relevant chunks in the Pinecone index, 
    which contains financial quarterly data for fiscal years 2023, 2024, and 2025.
    
    Args:
        query: Search query text
        year: Specific year to filter (e.g., "2023", "2024", "2025")
        quarter: List of quarters to filter (e.g., ["Q1", "Q2"])
        org: List of organizations to filter (e.g., ["NVIDIA", "AMD", "INTEL"])
    """
    top_k = 10
    chunks = query_pinecone(query, top_k, org=org, year=year, quarter=quarter)
    
    if not chunks:
        return "No relevant financial data found for the specified criteria."
    
    # Format the chunks for better readability
    contexts = "\n---\n".join([f'Chunk {i+1}: {chunk}' for i, chunk in enumerate(chunks)])
    
    return contexts


@tool("web_search")
def web_search(query: str):
    """
    Performs web search for latest financial information and market insights.
    Useful for getting recent news, market trends, and supplementary financial information
    that may not be in the quarterly reports database.
    """
    if not SERPAPI_KEY:
        return "Error: SERPAPI_KEY is not set in environment variables"
    
    # Configure SerpAPI parameters
    serpapi_params = {
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "q": query,
        "num": 5,
    }
    
    try:
        # Create search instance and get results
        search = GoogleSearch(serpapi_params)
        results = search.get_dict()
        
        # Check for error in response
        if "error" in results:
            return f"SerpAPI Error: {results['error']}"
        
        if "organic_results" not in results or not results["organic_results"]:
            return "No web search results found."
        
        # Format the search results
        contexts = "\n---\n".join([
            f"Title: {x.get('title', 'N/A')}\n"
            f"Summary: {x.get('snippet', 'N/A')}\n"
            f"Source: {x.get('link', 'N/A')}"
            for x in results["organic_results"]
        ])
        
        return contexts
        
    except Exception as e:
        return f"Web search error: {type(e).__name__}: {str(e)}"


@tool("final_answer")
def final_answer(
    research_steps: str,
    financial_analysis: str,
    market_insights: str,
    key_metrics: str,
    summary: str,
    sources: str
):
    """
    Compiles a comprehensive financial research report combining all gathered information.
    
    Args:
        research_steps: Bullet points detailing each research step taken
        financial_analysis: Detailed analysis from Pinecone vector search (2+ paragraphs)
        market_insights: Current market trends and news from web search (2+ paragraphs)
        key_metrics: Important financial metrics and KPIs extracted
        summary: Executive summary combining all findings (3+ paragraphs)
        sources: List of all referenced sources with links where available
    
    Returns:
        Structured financial research report
    """
    # Format lists if needed
    if isinstance(research_steps, list):
        research_steps = "\n".join([f"- {r}" for r in research_steps])
    if isinstance(sources, list):
        sources = "\n".join([f"- {s}" for s in sources])
    
    report = {
        "research_steps": research_steps or "No research steps documented",
        "financial_analysis": financial_analysis or "No financial analysis available",
        "market_insights": market_insights or "No market insights gathered",
        "key_metrics": key_metrics or "No key metrics identified",
        "summary": summary or "Unable to generate summary",
        "sources": sources or "No sources referenced"
    }
    
    return report


def init_research_agent(tool_keys, year=None, quarter=None, org=None):
    """Initialize the research agent with specified tools and context"""
    
    tool_str_to_func = {
        "web_search": web_search,
        "vector_search": vector_search,
        "final_answer": final_answer
    }
    
    # Always include final_answer, then add requested tools
    tools = [final_answer]
    for val in tool_keys:
        if val in tool_str_to_func:
            tools.append(tool_str_to_func[val])

    ## Designing Agent Features and Prompt ##
    system_prompt = f"""You are a Financial Research Agent specializing in analyzing quarterly reports 
    and financial data from fiscal years 2023, 2024, and 2025.
    
    Context:
    - Year: {year or 'Not specified'}
    - Quarter: {quarter or 'Not specified'}
    - Organization: {org or 'Not specified'}
    - Available Data: Quarterly financial reports stored in Pinecone vector database
    - Data Coverage: Fiscal years 2023, 2024, and 2025

    Your capabilities:
    1. Vector Search: Query the Pinecone database for specific financial data from quarterly reports
    2. Web Search: Get latest market news, trends, and supplementary financial information
    3. Analysis: Combine and analyze data from multiple sources to provide comprehensive insights

    Rules for tool usage:
    - If a tool has been used with a particular query, do NOT use it again with the same query
    - Do NOT use any tool more than twice in the same research session
    - Prioritize vector_search for historical quarterly data (2023-2025)
    - Use web_search for recent market trends and news
    - Always validate year/quarter parameters are within the available range (2023-2025)
    - Collect information from both sources when possible before providing final answer
    
    When analyzing financial data:
    - Focus on key metrics: Revenue, Net Income, EPS, Operating Margin, etc.
    - Identify trends across quarters and years
    - Highlight significant changes or anomalies
    - Provide context for financial performance
    
    Once you have collected sufficient information (stored in the scratchpad), 
    use the final_answer tool to compile a comprehensive report."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("assistant", "scratchpad: {scratchpad}"),
    ])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=OPENAI_API_KEY,
        temperature=0
    )

    def create_scratchpad(intermediate_steps: list[AgentAction]):
        research_steps = []
        for i, action in enumerate(intermediate_steps):
            if action.log != "TBD":
                research_steps.append(
                    f"Tool: {action.tool}, input: {action.tool_input}\n"
                    f"Output: {action.log}"
                )
        return "\n---\n".join(research_steps)

    oracle = (
        {
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"],
            "scratchpad": lambda x: create_scratchpad(
                intermediate_steps=x["intermediate_steps"]
            ),
        }
        | prompt
        | llm.bind_tools(tools, tool_choice="any")
    )
    return oracle


## Router and Parent Agent functions
def run_oracle(state: AgentState, oracle):
    """Execute the oracle to determine next tool to use"""
    print("Running oracle to determine next action...")
    print(f"Current intermediate_steps: {len(state.get('intermediate_steps', []))}")
    
    out = oracle.invoke(state)
    
    if not out.tool_calls:
        print("No tool calls made by oracle")
        return state
    
    tool_name = out.tool_calls[0]["name"]
    tool_args = out.tool_calls[0]["args"]
    
    print(f"Oracle selected tool: {tool_name}")
    
    action_out = AgentAction(
        tool=tool_name,
        tool_input=tool_args,
        log="TBD"
    )
    
    return {
        **state,
        "intermediate_steps": [action_out]
    }


def router(state: AgentState):
    """Route to the appropriate tool based on oracle's decision"""
    if isinstance(state["intermediate_steps"], list) and state["intermediate_steps"]:
        tool_name = state["intermediate_steps"][-1].tool
        print(f"Routing to tool: {tool_name}")
        return tool_name
    else:
        print("Router: Invalid format or empty steps, defaulting to final_answer")
        return "final_answer"


def run_tool(state: AgentState):
    """Execute the selected tool with appropriate parameters"""
    tool_str_to_func = {
        "web_search": web_search,
        "vector_search": vector_search,
        "final_answer": final_answer
    }
    
    # Get tool name and arguments from the last action
    tool_name = state["intermediate_steps"][-1].tool
    tool_args = state["intermediate_steps"][-1].tool_input

    # Add state context for vector_search
    if tool_name == "vector_search":
        tool_args = {
            **tool_args,
            "year": state.get("year"),
            "quarter": state.get("quarter"),
            "org": state.get("org")
        }
    
    print(f"Executing {tool_name}")
    
    try:
        # Run the tool
        out = tool_str_to_func[tool_name].invoke(input=tool_args)
        
        # For final_answer, convert dict to JSON string for better storage
        if tool_name == "final_answer" and isinstance(out, dict):
            out = json.dumps(out, indent=2)
        
        action_out = AgentAction(
            tool=tool_name,
            tool_input=tool_args,
            log=str(out)
        )
        
        return {
            **state,
            "intermediate_steps": [action_out]
        }
    except Exception as e:
        print(f"Error running tool {tool_name}: {str(e)}")
        # Return error state
        action_out = AgentAction(
            tool=tool_name,
            tool_input=tool_args,
            log=f"Error: {str(e)}"
        )
        return {
            **state,
            "intermediate_steps": [action_out]
        }


## Langraph - Designing the Graph
def create_graph(research_agent, year=None, quarter=None, org=None):
    """Create the LangGraph workflow for financial research"""
    
    tools = [
        vector_search,
        web_search,
        final_answer
    ]

    # Initialize the graph with AgentState
    graph = StateGraph(AgentState)

    # Add nodes to the graph
    graph.add_node("oracle", partial(run_oracle, oracle=research_agent))
    graph.add_node("web_search", run_tool)
    graph.add_node("vector_search", run_tool)
    graph.add_node("final_answer", run_tool)

    # Set the entry point
    graph.set_entry_point("oracle")

    # Add conditional routing from oracle
    graph.add_conditional_edges(
        source="oracle",
        path=router,
    )

    # Create edges from each tool back to oracle (except final_answer)
    for tool_obj in tools:
        if tool_obj.name != "final_answer":
            graph.add_edge(tool_obj.name, "oracle")

    # Final answer leads to END
    graph.add_edge("final_answer", END)

    # Compile the graph
    runnable = graph.compile()
    return runnable


def run_agents(query: str, tool_keys: List[str] = None, year: str = None, 
               quarter: List[str] = None, org: List[str] = None):
    """
    Main function to run the financial research agent
    
    Args:
        query: The research question or query
        tool_keys: List of tools to use (default: ["vector_search", "web_search"])
        year: Specific year to filter (e.g., "2023", "2024", "2025")
        quarter: List of quarters to filter (e.g., ["Q1", "Q2"])
        org: List of organizations to filter (e.g., ["NVIDIA", "AMD", "INTEL"])
    
    Returns:
        The compiled runnable graph
    """
    # Default tools if none specified
    if tool_keys is None:
        tool_keys = ["vector_search", "web_search"]
    
    # Validate year if provided
    if year and year not in ["2023", "2024", "2025"]:
        print(f"Warning: Year {year} not in available range (2023-2025)")
    
    # Initialize the research agent
    research_agent = init_research_agent(tool_keys, year, quarter, org)
    
    # Create and compile the graph
    runnable = create_graph(research_agent, year, quarter, org)
    
    # Create initial state
    initial_state = {
        "input": query,
        "chat_history": [],
        "intermediate_steps": [],
        "year": year,
        "quarter": quarter,
        "org": org
    }
    
    print(f"Starting financial research for query: {query}")
    print(f"Filters - Year: {year}, Quarter: {quarter}, Org: {org}")
    
    # Return the runnable for execution
    return runnable, initial_state


# Example usage
if __name__ == "__main__":
    # Example query
    query = "What were NVIDIA's Q2 2024 financial results and how do they compare to market expectations?"
    
    # Run the agent
    runnable, initial_state = run_agents(
        query=query,
        tool_keys=["vector_search", "web_search"],
        year="2024",
        quarter=["Q2"],
        org=["NVIDIA"]
    )
    
    # Execute the graph
    result = runnable.invoke(initial_state)
    
    # Print the final result
    if "intermediate_steps" in result:
        for step in result["intermediate_steps"]:
            if step.tool == "final_answer":
                print("\n=== FINAL REPORT ===")
                print(json.dumps(step.log, indent=2))