import uuid
from typing import TypedDict, Annotated, List
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv('Groq') # type: ignore
os.environ["TAVILY_API_KEY"] = os.getenv('Tavily') # type: ignore

class State(TypedDict): 
    linkedin_topic: str
    messages: Annotated[list, add_messages]
    human_feedback: str
    user_decision: str
    post_content: str  # Store the actual post content separately

@tool
def web_search(query: str) -> str:
    """Search the web for current information and news."""
    tavily = TavilySearchResults(max_results=3)
    result = tavily.invoke(query)
    formatted_results = []
    
    if isinstance(result, list): 
        for item in result:
            if isinstance(item, dict): 
                title = item.get('title', 'No Title')
                url = item.get('url', 'No URL')
                content = item.get('content', 'No Content')
                truncated_content = (content[:200] + '...') if len(content) > 200 else content
                formatted_results.append(f"Title: {title}\nURL: {url}\nContent: {truncated_content}\n---")
    
    return f"Search Results:\n" + "\n".join(formatted_results)

# Initialize tools and LLM
tools = [web_search]
tool_node = ToolNode(tools)
llm = ChatGroq(model="llama3-8b-8192", temperature=0.5)
llm_with_tool = llm.bind_tools(tools)

def generate_post(state: State):
    """Generate or regenerate LinkedIn post based on topic and feedback"""
    topic = state["linkedin_topic"]
    feedback = state.get("human_feedback", "")
    
    print(f"[DEBUG] Generating post for topic: {topic}")
    print(f"[DEBUG] With feedback: '{feedback}'")
    
    if feedback and feedback.strip() and feedback.lower() != "skip":
        prompt = f"""
        LinkedIn Topic: {topic}
        
        User feedback on previous version: {feedback}
        
        Based on this feedback, generate an improved LinkedIn post. Make it engaging, professional, and incorporate the user's suggestions. Include emojis and relevant hashtags.
        
        Generate ONLY the LinkedIn post content, no additional commentary.
        """
    else:
        prompt = f"""
        LinkedIn Topic: {topic}
        
        Generate a structured, engaging, and professional LinkedIn post around this topic. 
        Make it compelling and shareable with proper formatting, emojis, and hashtags.
        
        Generate ONLY the LinkedIn post content, no additional commentary.
        """
    
    response = llm.invoke([
        SystemMessage(content="You are an expert LinkedIn content writer who creates engaging, professional posts. Always generate actual post content, never just commentary."), 
        HumanMessage(content=prompt)
    ])
    
    post_content = response.content.strip()
    print(f"[DEBUG] Generated post (first 100 chars): {post_content[:100]}...")
    
    return { 
        "messages": [AIMessage(content=post_content)],
        "post_content": post_content,
        "human_feedback": "",  # Clear feedback after using it
        "user_decision": "",   # Clear decision
    }

def should_use_tool(state: State):
    """Determine if we should use search tool or go to review"""
    # For now, skip search and go directly to review
    # You can modify this later to detect when search is needed
    return "get_review_decision"

def get_review_decision(state: State):
    """Show post and ask for user decision - INTERRUPT POINT"""
    post_content = state.get("post_content", "")
    
    if not post_content:
        # Fallback to messages
        messages = state.get("messages", [])
        if messages:
            post_content = messages[-1].content
    
    if post_content and post_content.strip():
        print("\n" + "="*60)
        print("CURRENT LINKEDIN POST:")
        print("="*60)
        print(post_content)
        print("="*60)
        print("\nPost to LinkedIn? (yes/no):")
    else:
        print("[ERROR] No post content found!")
    
    return state

def route_after_review(state: State):
    """Route based on user decision"""
    decision = state.get("user_decision", "").lower().strip()
    print(f"[DEBUG] Routing with decision: '{decision}'")
    print(f"[DEBUG] Full state user_decision: '{state.get('user_decision', 'NOT_FOUND')}'")
    
    if decision and "yes" in decision:
        print("[DEBUG] Routing to POST")
        return "post"
    elif decision and ("no" in decision or decision):  # Any non-empty response that's not "yes"
        print("[DEBUG] Routing to COLLECT_FEEDBACK")
        return "collect_feedback"
    else:
        print("[DEBUG] Empty decision, defaulting to COLLECT_FEEDBACK")
        return "collect_feedback"

def collect_feedback(state: State):
    """Collect feedback from user - INTERRUPT POINT"""
    post_content = state.get("post_content", "")
    
    if not post_content:
        messages = state.get("messages", [])
        if messages:
            post_content = messages[-1].content
    
    if post_content:
        print("\n" + "="*60)
        print("CURRENT LINKEDIN POST:")
        print("="*60)
        print(post_content)
        print("="*60)
        print("\nHow can I improve this post? (or type 'skip' to regenerate without changes):")
    
    return state

def post(state: State):
    """Final posting step"""
    post_content = state.get("post_content", "")
    
    if not post_content:
        messages = state.get("messages", [])
        if messages:
            post_content = messages[-1].content
    
    if post_content:
        print("\n" + "="*60)
        print("FINAL LINKEDIN POST - NOW LIVE!")
        print("="*60)
        print(post_content)
        print("="*60)
        print("✅ Post has been approved and is now live on LinkedIn!")
    
    return {
        "messages": state["messages"] + [AIMessage(content="Post successfully published to LinkedIn!")]
    }

# Build the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("generate_post", generate_post)
workflow.add_node("search", tool_node) 
workflow.add_node("get_review_decision", get_review_decision)
workflow.add_node("collect_feedback", collect_feedback)
workflow.add_node("post", post)

# Set entry point
workflow.set_entry_point("generate_post")

# Add conditional edges with proper routing
workflow.add_conditional_edges(
    "generate_post",
    should_use_tool,
    {
        "search": "search",
        "get_review_decision": "get_review_decision"
    }
)

workflow.add_conditional_edges(
    "get_review_decision",
    route_after_review,
    {
        "post": "post",
        "collect_feedback": "collect_feedback"
    }
)

# Add regular edges
workflow.add_edge("search", "get_review_decision")
workflow.add_edge("collect_feedback", "generate_post")  # This is the key feedback loop!
workflow.add_edge("post", END)

# Compile with interrupts
checkpointer = MemorySaver()
app = workflow.compile(
    checkpointer=checkpointer, 
    # interrupt_after=[],
    interrupt_before=["get_review_decision","collect_feedback"]
)

def run_workflow():
    """Main function to run the LinkedIn post generator"""
    thread_config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    # Get topic from user
    linkedin_topic = input("Enter your LinkedIn topic: ")
    
    # Initial state
    initial_state = {
        "linkedin_topic": linkedin_topic,
        "messages": [],
        "human_feedback": "",
        "user_decision": "",
        "post_content": ""
    }
    
    print(f"\n🚀 Starting LinkedIn post generation for topic: '{linkedin_topic}'")
    
    try:
        # Start the workflow - this will run until first interrupt
        result = app.invoke(initial_state, config=thread_config)
        
        # Main interaction loop
        while True:
            # Get current state
            current_state = app.get_state(config=thread_config)
            
            if not current_state.next:
                print("\n🎉 Workflow completed!")
                break
                
            next_node = current_state.next[0] if current_state.next else None
            print(f"[DEBUG] Waiting at: {next_node}")
            
            if next_node == "get_review_decision":
                # Get user decision
                user_input = input().strip().lower()
                print(f"[DEBUG] User input for decision: '{user_input}'")
                
                # Update state with decision
                app.update_state(
                    config=thread_config,
                    values={"user_decision": user_input}
                )
                
                # Check the decision and handle it explicitly
                if "yes" in user_input:
                    print("[DEBUG] User chose YES - continuing to POST")
                    result = app.invoke(None, config=thread_config)
                    # Should complete the workflow
                    break
                else:
                    print("[DEBUG] User chose NO - continuing to FEEDBACK")
                    result = app.invoke(None, config=thread_config)
                
            elif next_node == "collect_feedback":
                # Get user feedback
                user_feedback = input().strip()
                print(f"[DEBUG] User feedback: '{user_feedback}'")
                
                # Update state with feedback and clear decision
                app.update_state(
                    config=thread_config,
                    values={"human_feedback": user_feedback, "user_decision": ""}
                )
                
                # Continue execution - this should loop back to generate_post
                result = app.invoke(None, config=thread_config)
                print(f"[DEBUG] After feedback, continuing workflow")
                
            else:
                print(f"[ERROR] Unexpected state: {next_node}")
                # Get the state for debugging
                debug_state = app.get_state(config=thread_config)
                print(f"[DEBUG] Full state: {debug_state.values}")
                break
                
    except KeyboardInterrupt:
        print("\n\n👋 Workflow cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

# Run the workflow
if __name__ == "__main__":
    run_workflow()