from pydantic import BaseModel,Field ,create_model                      #type:ignore
from typing import List,Optional
from dotenv import load_dotenv                                          #type:ignore
from enum import Enum

load_dotenv()





def get_first_critic_structure(subtopics: list[str]):
    
    TopicEnum = Enum("TopicEnum", {t: t for t in subtopics})

    
    class TopicJudgement(BaseModel):
        
        topic: TopicEnum = Field(description="The subtopic name")
        approval: bool
        opinion: str

    
    class f_critic_structure (BaseModel):
        judgements : List[TopicJudgement] = Field(description="List of topic judgments")
    
    return f_critic_structure



def parse_first_critic_output(output):
    ca={}
    co={}
    for j in output.judgements :
        
        topic = getattr(j.topic, "value", j.topic)
        ca[topic] = j.approval
        co[topic] = j.opinion
    return ca , co




class searcher_structure(BaseModel):

    subtopics: List[str]
 

class f_summarizer_structure(BaseModel):
    summary  : str = Field(description="Put your summary here")
    KeyFacts : List[str]  = Field(description="Here, list the key points and facts from all the lists you were given, don't list too many nor too few")

    
# class f_critic_structure(BaseModel):
#     approval_booleans : List[bool] = Field(description="A list of booleans, which describes if the critic AI approved the summary. The critic should put True if it is approved, and False if not. Important: It should put the booleans in THE SAME ORDER AS THE ORDER OF THE summaries IN THE GIVEN LIST")
#     opinions : List[str] = Field(description="Here, The critic AI should put the reasons why it didn't approve a summary if fit didn't, if the summary was approved, te critic's opinion should be 'None'. Important: It should put the booleans in THE SAME ORDER AS THE ORDER OF THE summaries IN THE GIVEN LIST")

class final_critic_structure(BaseModel):
    approval:bool = Field(description = "a boolean, put True if you approveand accept the summary, and pu False otherwise")
    opinion:str = Field(description = "your opinion and reasons of rejecting the summary if you did reject it. Important: If the summary was approved then you should put 'None' here")
