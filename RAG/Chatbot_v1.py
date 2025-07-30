from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END # type: ignore
from langgraph.graph.message import add_messages # type: ignore
from langchain_tavily import TavilySearch  # type: ignore
from langgraph.checkpoint.memory import MemorySaver # type: ignore
from langchain_groq import ChatGroq # type: ignore
from langchain_core.messages import HumanMessage, AIMessage # type: ignore
from langchain_core.messages import ToolMessage # type: ignore
from langchain_core.tools import InjectedToolCallId, tool # type: ignore
from langgraph.prebuilt import ToolNode, tools_condition  # type: ignore
from langgraph.types import Command, interrupt # type: ignore
import os
import json
from dotenv import load_dotenv # type: ignore


# Set environment variables
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LangChain') # type: ignore
os.environ["GROQ_API_KEY"] = os.getenv('Groq') # type: ignore
os.environ["TAVILY_API_KEY"] = os.getenv('Tavily') # type: ignore


class State(TypedDict):
    messages: Annotated[list, add_messages]

# Initialize the chat model with Groq
llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0.5,
)

@tool # type: ignore
def human_assistance(query: str,source: str,tool_call_id: Annotated[str, InjectedToolCallId]) -> str:
    """Request assistance from a human."""
    try:
        human_response = interrupt(
            {
                "question": "Is this correct?",
                "query": query,
                "source": source
            },
        )
        if human_response.get("correct", "").lower().startswith("y"):
            verified_query = query
            verified_source = source
            response = "Correct"
        else:
            verified_query = human_response.get("query", query)
            verified_source = human_response.get("birthday", source)
            response = f"Made a correction: {human_response}"
        return response
    except Exception as e:
        return f"Human validation error: {str(e)}"

# Initialize tools
search_tool = TavilySearch(max_results=2)
tools = [search_tool,human_assistance] 

graph_builder = StateGraph(State)

# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)
memory = MemorySaver()

def chatbot(state: State):
    try:
        message = llm_with_tools.invoke(state["messages"])
        return {"messages": [message]}
    except Exception as e:
        error_message = AIMessage(content=f"Error in chatbot: {str(e)}")
        return {"messages": [error_message]}

# class BasicToolNode:
#     """A node that runs the tools requested in the last AIMessage."""

#     def __init__(self, tools: list) -> None:
#         self.tools_by_name = {tool.name: tool for tool in tools}

#     def __call__(self, inputs: dict):
#         if messages := inputs.get("messages", []):
#             message = messages[-1]
#         else:
#             raise ValueError("No message found in input")
#         outputs = []
#         for tool_call in message.tool_calls:
#             tool_result = self.tools_by_name[tool_call["name"]].invoke(
#                 tool_call["args"]
#             )
#             outputs.append(
#                 ToolMessage(
#                     content=json.dumps(tool_result),
#                     name=tool_call["name"],
#                     tool_call_id=tool_call["id"],
#                 )
#             )
#         return {"messages": outputs}


def route_tools(state: State,):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    try:
        if isinstance(state, list):
            ai_message = state[-1] # type: ignore
        elif messages := state.get("messages", []):
            ai_message = messages[-1]
        else:
            raise ValueError(f"No messages found in input state: {state}")
        
        if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0: # type: ignore
            return "tools"
        return END
    except Exception as e:
        print(f"Error in route_tools: {e}")
        return END


# tool_node = BasicToolNode(tools=[tools])
tool_node = ToolNode(tools=tools)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", tool_node)
graph_builder.add_conditional_edges(
    "chatbot",
    route_tools,
    {"tools": "tools", END: END},
)
# Any time a tool is called, we return to the chatbot to decide the next step
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer=memory,interrupt_before=["tools"])

# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

config = {"configurable": {"thread_id": "1"}}

def stream_graph_updates(user_input: str):
    try:
        events = graph.stream(
            {"messages": [HumanMessage(content=user_input)]},
            config,
            stream_mode="values",
        )
        
        for event in events:
            if "messages" in event and event["messages"]:
                last_message = event["messages"][-1]
                print("="*100)
                if hasattr(last_message, 'content'):
                    print("Assistant: ", last_message.content)
                else:
                    print("Assistant: ", str(last_message))
                print("="*100)   
    except Exception as e:
        print(f"Error in stream_graph_updates: {e}")
        import traceback
        traceback.print_exc()

def extract_conversation_history(snapshot):
    """Extract and format conversation history from StateSnapshot"""
    messages = snapshot.values.get('messages', [])
    conversation_history = []
    
    for message in messages:
        try:
            if isinstance(message, HumanMessage):
                conversation_history.append({
                    'role': 'user',
                    'content': message.content,
                    'timestamp': getattr(message, 'id', 'unknown')
                })
            elif isinstance(message, AIMessage):
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    conversation_history.append({
                        'role': 'assistant',
                        'content': '[Tool Call]',
                        'tool_calls': message.tool_calls,
                        'timestamp': getattr(message, 'id', 'unknown')
                    })
                else:
                    conversation_history.append({
                        'role': 'assistant',
                        'content': message.content,
                        'timestamp': getattr(message, 'id', 'unknown')
                    })
            elif isinstance(message, ToolMessage):
                conversation_history.append({
                    'role': 'tool',
                    'content': f"[Tool Response: {message.name}]",
                    'tool_name': message.name,
                    'timestamp': getattr(message, 'id', 'unknown')
                })
        except Exception as e:
            print(f"Error processing message: {e}")
            continue
    
    return conversation_history

def handle_human_validation():
    """Handle human validation for interrupted tool calls"""
    try:
        # Get current state
        snapshot = graph.get_state(config)
        
        if not snapshot.next:
            print("No pending interrupts found.")
            return False
            
        print(f"Current interrupts: {snapshot.next}")
        
        # Get the last message to find tool calls
        last_message = snapshot.values.get('messages', [])[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            tool_call = last_message.tool_calls[0]  # Get first tool call
            print(f"Tool call: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Get human input for validation
            query = input("Human validation (query): ")
            source = input("Human validation (source if any): ")
            correct = input("Is this correct? (y/n): ")
            
            # Resume with human input
            human_input = {
                "query": query,
                "source": source,
                "correct": correct
            }
            
            # Resume the graph with the human input
            graph.update_state(config, human_input)
            
            # Continue execution
            events = graph.stream(None, config, stream_mode="values")
            for event in events:
                if "messages" in event and event["messages"]:
                    last_message = event["messages"][-1]
                    print("="*100)
                    if hasattr(last_message, 'content'):
                        print("Assistant: ", last_message.content)
                    else:
                        print("Assistant: ", str(last_message))
                    print("="*100)
            
            return True
            
    except Exception as e:
        print(f"Error in handle_human_validation: {e}")
        import traceback
        traceback.print_exc()
        return False


    
while True:
    try:
        user_input = input("User: ")
        
        if user_input.lower() in ["quit", "exit", "q"]:
            print("-"*100)
            print("Saving conversation history to conversation_history.json")
            snapshot = graph.get_state(config)
            conversation_data = extract_conversation_history(snapshot)
            with open('conversation_history.json', 'w') as f:
                json.dump(conversation_data, f, indent=2)
            break
        
        # Process the user input
        stream_graph_updates(user_input)
        
        # Check for interrupts and handle human validation
        snapshot = graph.get_state(config)
        if snapshot.next:
            print(f"Interrupt detected: {snapshot.next}")
            if not handle_human_validation():
                print("Failed to handle human validation. Continuing...")
                
    except KeyboardInterrupt:
        print("\nExiting...")
        break
    except Exception as e:
        print(f"Error in main loop: {e}")
        import traceback
        traceback.print_exc()
        break
