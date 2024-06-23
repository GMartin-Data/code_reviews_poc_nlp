"""
Source code for a basic agent, able to expand its capabilities
with browsing the Web using a tool
"""

# ENVIRONMENT: Set all that the agent will need
## SOURCE CREDENTIALS
from dotenv import load_dotenv

load_dotenv()  # API keys: LLMs, tools...


## AGENT STATE
import operator
from typing import Annotated

from langchain_core.messages import AnyMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """Records the agent's state along time"""
    messages: Annotated[list[AnyMessage], operator.add]
    

## TOOLS
from langchain_community.tools.tavily_search import TavilySearchResults

tools = [TavilySearchResults(max_results=4)]


## CHECKPOINTER (Agent's memory)
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")


## PROMPT
prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""


## LLM
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o")


# AGENT
from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from rich import print as rprint

class Agent:
    def __init__(self, model, tools: list, checkpointer, system: str = "") -> None:
        # Store system prompt
        self.system = system
        # Build and compile graph, then store it as an attribute
        graph_builder = StateGraph(AgentState)
        ## Nodes
        graph_builder.add_node("llm", self.call_openai)
        graph_builder.add_node("action", self.take_action)
        ## Edges
        graph_builder.add_conditional_edges(
            "llm",  # Initial node
            self.exists_action,  # Decision function
            # Map decision function output to destination node
            {True: "action", False: END}
        )
        graph_builder.add_edge("action", "llm")
        ## Set entry point, compile and store
        graph_builder.set_entry_point("llm")
        self.graph = graph_builder.compile(checkpointer=checkpointer)
        ## Convenience mapping for tools
        self.tools = {tool.name for tool in tools}
        ## Bind tools to the model
        self.model = model.bind_tools(tools)
    
    ## NODES METHODS
    def call_openai(self, state: AgentState) -> AgentState:
        """Defines the method called within the 'llm' node"""
        messages = state["messages"]
        if self.system:
            # prepend system message
            messages = [SystemMessage(content=self.system)] + messages
        new_message = self.model.invoke(messages)
        return {"messages": [new_message]}
    
    def take_action(self, state: AgentState) -> AgentState:
        """Defines the method called within the 'action' node"""
        rprint("[bold red]To asnwer this, I need to browse the Web...[/bold red]")
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool in tool_calls:
            rprint(f"Calling {tool}")
            # Retrieve tool result
            tool_result = self.tools[tool["name"]].invoke(tool["args"])
            # Transmit corresponding ToolMessage
            results.append(
                ToolMessage(
                    tool_call_id = tool["id"],
                    name = tool["name"],
                    content = str(tool_result)
                )
            )
        rprint("[bold blue]Tool results transmitted to the model[/bold blue]")
        return {"messages": results}        
        
    ## CONDITIONAL EDGE METHOD
    def exists_action(self, state: AgentState) -> bool:
        """
        Decision function to call the Web tool or not.
        Checks if a tool call exists within the last emitted message from the state.
        """
        last_message = state["messages"][-1]
        return len(last_message.tool_calls) > 0
    

agent = Agent(llm, tools, system=prompt, checkpointer=memory)
    
    
if __name__ == "__main__":
    from langchain_core.messages import AIMessage, HumanMessage
    
    rprint("Conversational Chatbot - When done type quit, exit or q, then ENTER")
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ("quit", "exit", "q"):
            rprint("Goodbye!")
            break
        for event in agent.graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            {"configurable": {"thread_id": "1"}}
        ):
            for value in event.values():
                message = value["messages"][-1]
                if isinstance(message, AIMessage) and message.content:
                    rprint(message.content)    
