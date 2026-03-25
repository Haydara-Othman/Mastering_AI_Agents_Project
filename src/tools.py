import PyPDF2                       #type:ignore
import os
from dotenv import load_dotenv      #type:ignore
from langchain_tavily import TavilySearch                               #type:ignore


load_dotenv()

def get_tavily_tool():
    tavily_api = os.getenv("TAVILY_API_KEY", None)

    if not tavily_api is None:
        tavily_tool=TavilySearch(tavily_api_key=tavily_api ,max_results=6 , search_depth="advanced")
        return tavily_tool
    else :
        raise ValueError(
                    "No valid Tavily API key found. Please set one in your .env file then try again"
                )

def read_txt_file(path):
    with open(path , 'r') as f:
        return f.read()

def read_pdf_file(path):

    with open(path, "rb") as pdf_file:
        reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()

    return text

def load_documents():
    
    results = []
    docs =[ x  for x in os.listdir("data") ]
    names=[doc[:-4] for doc in docs]
    for name,doc in zip(names,docs) :
        path = os.path.join("data" , doc)
        if doc.endswith(".txt"):
            results.append({'content':read_txt_file(path) , 'title':name})
        elif doc.endswith(".pdf"):
            results.append({'content':read_pdf_file(path) , 'title':name})
    return results

def  add_rag_to_tavily_results(state):
    """Augment Tavily search results with local RAG hits from the vectordb."""
    tr = state["search_results"]
    vdb = state["vectordb"]

    if vdb is None:
        return tr

    for topic, results in tr.items():
        rag_results = vdb.search(topic, n_results=20)
        rag_contents = [result["content"] for result in rag_results]
        # Concatenate original Tavily paragraphs with RAG paragraphs
        tr[topic] = list(results) + rag_contents

    return tr
    

