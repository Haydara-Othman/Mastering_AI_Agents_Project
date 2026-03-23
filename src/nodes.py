from langchain_groq import ChatGroq                                     #type:ignore
from langchain_google_genai import ChatGoogleGenerativeAI               #type:ignore
from langchain_openai import ChatOpenAI                                 #type:ignore
from langchain_tavily import TavilySearch                               #type:ignore
from prompts import simplifier_prompt  , first_summarizer_prompt , first_critic_prompt , final_summarizer_prompt ,final_critic_prompt ,first_critic_list_template
from typing_extensions import TypedDict                                 #type:ignore
from typing import List,Optional,Dict
from dotenv import load_dotenv                                          #type:ignore
import os
from tools import load_documents,add_rag_to_tavily_results
from vectordb import VectorDB
from output_structures import searcher_structure, f_summarizer_structure , final_critic_structure  , get_first_critic_structure , parse_first_critic_output


load_dotenv()
tavily_api = os.getenv("TAVILY_API_KEY", None)

if not tavily_api is None:
    tavily_tool=TavilySearch(tavily_api_key=tavily_api ,max_results=6 , search_depth="advanced")
else :
    raise ValueError(
                "No valid Tavily API key found. Please set one in your .env file then try again"
            )



def _initialize_llm():
        
        
        if os.getenv("OPENAI_API_KEY"):
            model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            print(f"Using OpenAI model: {model_name}")
            return ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"), model=model_name, temperature=0.0
            )

        elif os.getenv("GOOGLE_API_KEY"):
            model_name = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
            print(f"Using Google Gemini model: {model_name}")
            return ChatGoogleGenerativeAI(
                google_api_key=os.getenv("GOOGLE_API_KEY"),
                model=model_name,
                temperature=0.0,
            )
            
        elif os.getenv("GROQ_API_KEY"):
            model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
            print(f"Using Groq model: {model_name}")
            return ChatGroq(
                api_key=os.getenv("GROQ_API_KEY"), model=model_name, temperature=0.0
            )

        else:
            raise ValueError(
                "No valid API key found. Please set one of: OPENAI_API_KEY, GROQ_API_KEY, or GOOGLE_API_KEY in your .env file"
            )


llm = _initialize_llm()










class State(TypedDict):
    main_topic : str 
    sub_topics : List[str]
    search_results : Dict
    search_with_RAG_results :Dict
    topics_summaries : Dict
    topics_kfacts    : Dict
    summaries_approval : Dict
    first_critic_opinions : Dict
    final_summary : str
    final_critic_approval : bool
    final_critic_opinion : str
    vectordb : Optional[VectorDB] = None
    num_first_critics : int
    num_final_critics : int
    num_external_resources : int





simplifier_llm = llm.with_structured_output(searcher_structure)
summarizer_1_llm = llm.with_structured_output(f_summarizer_structure)
critic_2_llm = llm.with_structured_output(final_critic_structure)

searcher_chain = simplifier_prompt | simplifier_llm
first_summarizer_chain   = first_summarizer_prompt   | summarizer_1_llm
final_summarizer_chain = final_summarizer_prompt | llm
final_critic_chain = final_critic_prompt | critic_2_llm










def Searcher(state):
    print("[Searcher] starting")
    print(f"[Searcher] main_topic = {state['main_topic']!r}")
    

    response = searcher_chain.invoke(
        state['main_topic']
    )

    dc=state["search_results"]
    
    if getattr(response, "subtopics", None):
        subtopics=response.subtopics
        for topic in subtopics:
            print(f"[Searcher] querying Tavily for subtopic: {topic}")
            results = tavily_tool.invoke(topic)
            results_contents = [result['content'] for result in results.get('results', [])]
            print(f"[Searcher] got {len(results_contents)} web results for {topic}")
            dc[topic] = results_contents
    else:
        subtopics=[]
        print("[Searcher] WARNING: no subtopics were returned by the LLM")

    print(f"[Searcher] done. sub_topics = {list(response.subtopics) if getattr(response, 'subtopics', None) else []}")
    return {
        'search_results' : dc,
        'sub_topics': subtopics,
        'summaries_approval' : {s :False for s in subtopics},
        'first_critic_opinions' : {s :'None' for s in subtopics}

    }
        
def VDB_builder(state):
    print("[VDB_builder] starting")
    vdb=VectorDB()
    docs=load_documents()
    vdb.add_documents(docs)
    print(f"[VDB_builder] loaded {len(docs)} local documents")
    return {"vectordb" : vdb   , 'num_external_resources' : len(docs)} 

def Single_Summarizer(topic,results,opinion):
    print(f"[Single_Summarizer] summarizing topic={topic!r} with {len(results)} paragraphs")
    results_string = "\n\n\n\n\n".join(results)
    response = first_summarizer_chain.invoke({'topic': topic, 'paragraphs_list': results_string, 'opinion': opinion})
    return response

def Parallel_Analyzer(state):

    print("[Parallel_Analyzer] starting")
    
    if state['num_external_resources'] > 0:
        print(f"[Parallel_Analyzer] num_external_resources = {state['num_external_resources']}, adding RAG to Tavily results")
        sr = add_rag_to_tavily_results(state)
    else:
        print("[Parallel_Analyzer] no external resources, using pure Tavily search_results")
        sr = state['search_results']

    ts = state["topics_summaries"]
    tkf= state["topics_kfacts"]
    print(f"[Parallel_Analyzer] have {len(sr)} topics to process")

    for topic,results in sr.items():
        
        skip = state["summaries_approval"][topic]
        if skip:
            continue
        
        opinion = state["first_critic_opinions"][topic]
        response = Single_Summarizer(topic, results, opinion)
        ts[topic]  = getattr(response, "summary", 'None')
        tkf[topic] = getattr(response, "KeyFacts", [])


    print(f"[Parallel_Analyzer] done. topics_summaries topics = {list(ts.keys())}")
    return {'topics_summaries' : ts   ,  'topics_kfacts' : tkf}

def First_Critic(state):

    print("[FirstCritic] starting")
    
    topics = state["sub_topics"]
    ts = state["topics_summaries"]
    tkf= state["topics_kfacts"]

    structure = get_first_critic_structure (topics)
    critic_1_llm = llm.with_structured_output(structure)
    first_critic_chain = first_critic_prompt | critic_1_llm


    ts_list=[ first_critic_list_template.format(**{'topic' : t , 'summary' : s , 'key_facts' : tkf[t]}) for t,s in ts.items()]
    ts_string = "[ \n"    +    "\n\n\n\n".join(ts_list)    +   "\n ]"

    co=first_critic_chain.invoke(ts_string)
    approval_dict , opinion_dict =parse_first_critic_output(co)
    

    return {'summaries_approval' : approval_dict  , 'first_critic_opinions' : opinion_dict ,'num_first_critics' : state["num_first_critics"]+1 }

def Final_Summarizer(state):

    print("[Final_Summarizer] starting")
    
    ts=state["topics_summaries"]
    ts_list=[t + ":\n" + s for t,s in ts.items()]
    ts_string = "[ \n" + "\n\n\n\n".join(ts_list) + "\n ]"
    
    fs = final_summarizer_chain.invoke({'articles_list': ts_string, 'opinion': state["final_critic_opinion"]})
    text = fs.content if hasattr(fs, 'content') else str(fs)

    print(f"[Final_Summarizer] done. summary length = {len(text)} chars")
    return {'final_summary' : text}
    
def Final_Critic(state):
    print("[Final_Critic] starting")
    ts=state["topics_summaries"]
    ts_list=[t + ":\n" + s for t,s in ts.items()]
    ts_string = "[ \n" + "\n\n\n\n".join(ts_list) + "\n ]"
    result=final_critic_chain.invoke({'summary': state["final_summary"], 'summarizatoins_list': ts_string})
    print(f"[Final_Critic] approval = {result.approval}")
    return {'final_critic_approval':result.approval ,  'final_critic_opinion' : result.opinion , 'num_final_critics' : state["num_final_critics"]+1}
    




    
