from dotenv import load_dotenv                                          #type:ignore
from build_graph import build_graph

   


# Load environment variables
load_dotenv()


agent = build_graph()


while True:
    print("-"*50)
    t=input("Enter the topic you want a research about? ")
    print("-"*50)

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

    print("[main] invoking agent graph")
    result = agent.invoke(init_state)
    print("[main] agent run finished")

    print(result["final_summary"])






    









