import sys
from referee.__main__ import main
import random

type_of_agent = [
    "agent.FreckerPlayer:GreedyAgent",
    "agent.FreckerPlayer:SmarterRandomPlayer",
    "agent.FreckerPlayer:SlowMiniMaxAgent",
    "agent.FreckerPlayer:MiniMaxAgent",
    "agent.FreckerPlayer:InEfficientMiniMaxAgent",
    "agent.FreckerPlayer:CorrectMiniMaxAgent",
    "agent.FreckerPlayer:MLMiniMaxAgent"
]
main_agent = "agent.MCTS_XG.agent:MCTS_Agent"

TIMES = 10

for i in range(TIMES):
    agent = random.choice(type_of_agent)
    print(f"Running test {i+1} with agent: {agent}")
    if i % 2 == 0:
        sys.argv = ["referee", main_agent, agent]
    else:
        sys.argv = ["referee", agent, main_agent]
    try:
        main()
    except SystemExit:
        pass  # Prevent script from exiting