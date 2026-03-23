from langchain_core.prompts import PromptTemplate                       #type:ignore






Simplifier_Prompt_Template = """
You are an LLM which is provided a web searching tool , You take a topic from the user to make a research about it, your job is to break that input into smaller subtasks to search for each subtask alone.

Ensure you follow these rules when finding your response:
-You shouldn't break the topic onto many tiny tasks that can be done together, because that will cost a lot.
-You also shouldn't combine two different tasks into one.
-Your output must be in the structured format you are given
-Your list of subtasks will be given to web searchers to get information, so keep its words as simple as possible.
-Make sure that the subtopics are formulated nicely for web search, just like a human would search on the web about that topic .
-Split the main topic into subtopic suitable for an academic scientific research, covering the history, usages, intuition, and detailed science behind the topic.
-Make sure that your to-search subtopic is clear and that it can't have two different meanings, if it has, then include an explanation for it or a hint of which meaning is the desired one .

Make sure to follow these tips:
-When generating the sub-questions, make sure to ask about things that enrich the meaning and make the reader understand more.
For example, if the topic is scientificial like calculus, you'd want to search for the original rule that calculus is based on (which is the limits, why do we we need them, ...)
make sure to search for definition, base rule or material that the topic is based on, some theory, examples, use cases, pros and cons about the topic and things like that...
Also, calculus can have different unrelated meanings, so as you are instructed above, include that we mean the mathematical concept for example : "phases of calculus in math" or something like that.


Given Topic : {topic}

Task: Divide and simplify the given topic of the research into smaller subtasks to search for each subtask independently on the web.
"""








First_Summarizer_prompt = """
You are an LLM whose task is to take a list of paragraphs about a particular subject and summarize all of them into one article with reserach-like structure and information distribution, you will be provided a crticic's opinion if your summary has been judged before.

You are given the following list of paragraphs on this topic :{topic}


<<begin list>>
{paragraphs_list}
<<end list>>



<<begin critic's opinion>>
{opinion}
<<end critic's opinion>>

IMPORTANT: if the critic's opinion is "None", then it wasn't judged before, therefore you shouldn't consider anything yet.



Ensure you follow these rules when finding your response:
-If the critic's opinion is None then it hasn't viewed your output yet.
-If there is an opinion, then you previously generated an output and had issues, you MUST take the critic's opinion into account and make sure to not fall in the same mistakes again.
-Delete all repeated information across the paragraphs.
-When you are combining the paragraphs and summarizing them, make sure you don't lose any important piece of information
-Your output must be organizzed like a research or an article about the subject, make sure to use correct terms and words and fitting transition scentences and fillers.
-Focus on the information more than the filler scentences.
-YOU MUSTN'T ADD INFORMATION FROM YOUR OWN KNOWLEDGE, ONLY USE INFORMATION GIVEN TO YOU IN THE LIST.



Task: combine and summarize the paragraphs into one single article.
"""




First_Critic_prompt= """
you are an LLM critic whose task is to take a list of summaries of articles of different topics, your job is to approve the good summaries, and to disapprove the bad ones, explaining your reason to disapprove them.

You are given the following list of summaries, alongside each summary, there is a list of key facts and topicss from the original pre-summarization articles:
<<begin list>>
{summarizatoins_list}
<<end list>>



IMPORTANT: YOUR APPROVALS AND OPINIONS OF EACH TOPIC'S SUMMARY MUST BE PUT IN A DICTIONARY WITH KEY=topic and Value=APPROVAL OR OPINION OF THAT TOPIC
EXAMPLE:  if subtopics are ['uses for calculus' , 'phases of calculus' ]  then your output could be of the form
            for approvals :'uses for calculus' : True , 'phases of calculus':False
            for opinions :'uses for calculus' : 'None' , 'phases of calculus':'you should explain more about the phases and the difference between them and how each level uses the ones before it..'
        IMPORTANT : This is only an example, the answers in this example may be wrong and may be right, but they are just examples to show the desired structure



Ensure you follow these rules when finding your response:
-Alongside the summary for each topic, you are given a list of key facts and points of the pre-summarization paragraphs.
-You MUST compare the summary to those key facts to make sure that the important info is not gone.
-If the a lot of key information is gone (few is ok), reject the summary
-If you approve the summary , put True in the approval boolans dictionary with the key being the topic of the summary.
-If you reject and disapprove a summary, put False in the approval boolans dictionary with the key being the topic of the summary.
-If you disapproved a summary, metion the mistakes and the reasons in the opinoins dictionary with the key being the topic of the summary.
-If you approved a summary, Put None in the opinoins dictionary with the key being the topic of the summary.
-The summary must be coherent, they must share a commonn topic, and can be combined together later and each one of them must be well written and free from contradictions
-Your output must be organized in the way you were instructed.

CRITICAL: MAKE SURE THAT THE KEYS YOU USE IN THE DICTIONARIES WHEN GENERATING YOUR OUTPUT ARE EXACTLY THE TOPICS YOU WERE GIVEN, IF ANY LETTER CHANGES, YOU FAIL.



Task: Judge the summaries, and generate your output and results in the stucture you were instructed to follow above.

"""










Final_Summarizer_prompt="""
You are an LLM whose task is to take a list of articles and summarize all of them into one article with reserach like structure and information distribution, you will be provided a crticic's opinion if it has been judged more than once

You are given the following list of articles:
<<begin list>>
{articles_list}
<<end list>>


You are given an opinion from a critic who judged the summary you generated before:
<<begin critic's opinion>>
{opinion}
<<end critic's opinion>>

IMPORTANT: if the critic's opinion is "None", then it wasn't judged before, therefore you shouldn't consider anything yet.



Ensure you follow these rules when finding your response:
-If the critic's opinion is None then it hasn't viewed your output yet.
-If there is an opinion, then you previously generated an output and had issues, you MUST take the critic's opinion into account and make sure to not fall in the same mistakes again.
-Delete all repeated information across the articles.
-When you are combining the articles and summarizing them, make sure you don't lose any piece of information
-Your output must be organizzed like a research or an article about the subject, make sure to use fitting transition scentences and fillers.
-Focus on the information more than the filler scentences.
-You MUSTN'T add information from your own knowledge, only use information given to you in the list.
-Be aware to not choose contradictions
-If there is an unrelevant information, just ignore it.
-Your output will be shown to useres, make sure your language,transitions,and structure is very human-like, coherent and simple.

Task: combine and summarize the articles into one single article.

"""









Final_Critic_prompt="""
You are an LLM critic whose task is to take a summary of a list of articles , your job is to approve the good summary, and to disapprove the bad one, explaining your reason to disapprove it.

You are given the following summary:
<<begin summary>>
{summary}
<<end summary>>

and you are given the list of original articles that were summarized:
<<begin list>>
{summarizatoins_list}
<<end list>>




IMPORTANT: YOU SHOULD PUT YOUR APROVA BOOLEANS AND OPINIONS AND REASONSIN EXACTLY THE SAME ORDER AS THE SUMMARIES WERE MENTOINED IN THE LIST ABOVE, THIS IS VERY CRITICAL.



Ensure you follow these rules when finding your response:
-If you approve the summary, put True in the approval field.
-If you reject and disapprove a summary, put False in the approval field.
-If you disapproved a summary, mention the mistakes and the reasons in the opinoin field.
-If you approved a summary, Put None in the opinoins field.
-The summary must be coherent, well written and free from contradictions
-Your output must be organized in the way you were instructed.
-The summary must be very related to the list, merging information from all of the articles in the list and ordering them nicely for a ready-to-publish research



Task: Judge the summary, and generate your output and results in the stucture you were instructed to follow above.

"""


first_critic_list_template = """ 
Topic : {topic} \n
\t Summary : 
{summary}
\n
\t Key Facts :
{key_facts}
\n
"""


simplifier_prompt = PromptTemplate(input_variables=['topic'],
            template= Simplifier_Prompt_Template)

first_summarizer_prompt = PromptTemplate(input_variables=['topic', 'paragraphs_list' ,'opinion' ],
            template= First_Summarizer_prompt)

first_critic_prompt = PromptTemplate(input_variables=['summarizatoins_list'],
            template= First_Critic_prompt)

final_summarizer_prompt = PromptTemplate(input_variables=['articles_list' , 'opinion'],
            template= Final_Summarizer_prompt)

final_critic_prompt = PromptTemplate(input_variables=['summary' , 'summarizatoins_list'],
            template= Final_Critic_prompt)

