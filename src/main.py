from dotenv import load_dotenv                                          #type:ignore
from build_graph import build_graph
import asyncio

   


# Load environment variables
load_dotenv()


agent = build_graph()


while True:
    print("-"*50)
    t=input("Enter the topic you want a research about? (Type 'quit' to exit)")
    print("-"*50)
    if t.lower() =='quit':
        break

    init_state = {
        'main_topic' : t,
        'sub_topics' : [],
        'search_results' : {},
        'topics_summaries' : {},
        'topics_kfacts' : {},
        'summaries_approval' : {},
        'first_critic_opinions' : {},
        'final_summary' : '',
        'final_critic_approval' : False,
        'final_critic_opinion' : 'None',
        'vectordb' : None,
        'num_first_critics' : 0,
        'num_final_critics' : 0,
        'num_external_resources' :0

    }

    result = asyncio.run(agent.ainvoke(init_state))

    print(result["final_summary"])






    









