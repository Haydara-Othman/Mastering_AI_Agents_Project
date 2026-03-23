from langchain_groq import ChatGroq                                     #type:ignore
from langchain_google_genai import ChatGoogleGenerativeAI               #type:ignore
from langchain_openai import ChatOpenAI                                 #type:ignore
from langgraph.graph import StateGraph, END                             #type:ignore
from dotenv import load_dotenv                                          #type:ignore


from nodes import State, Searcher , VDB_builder, Parallel_Analyzer, First_Critic, Final_Summarizer , Final_Critic

load_dotenv()









def first_summarizer_edges(state):
    if state["num_first_critics"] >3 or all(state["summaries_approval"].values())  :
        return "Final_Summarizer"
    
    return "First_Critic"

def final_summarizer_edges(state):
    if state["num_final_critics"] > 3 or state["final_critic_approval"]:
        return END
    
    return "Final_Critic"





def build_graph():

    graph = StateGraph(State)
        
        
    graph.add_node("Searcher", Searcher)
    graph.add_node("VDB_builder", VDB_builder)
    graph.add_node("Parallel_Analyzer", Parallel_Analyzer)
    graph.add_node("First_Critic", First_Critic)
    graph.add_node("Final_Summarizer", Final_Summarizer)
    graph.add_node("Final_Critic", Final_Critic)



    graph.set_entry_point("Searcher")

    graph.add_edge("Searcher", "VDB_builder")  
    graph.add_edge("VDB_builder", "Parallel_Analyzer")  
    graph.add_conditional_edges("Parallel_Analyzer", first_summarizer_edges, {"Final_Summarizer": "Final_Summarizer", "First_Critic": "First_Critic"})
    graph.add_edge("First_Critic", "Parallel_Analyzer")    
    graph.add_conditional_edges("Final_Summarizer", final_summarizer_edges, {"Final_Critic": "Final_Critic", END: END})
    graph.add_edge("Final_Critic", "Final_Summarizer")

    agent=graph.compile()

    return agent


