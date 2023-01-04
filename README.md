#########################################################################################
# Submission By: Aditya Bhushan
#########################################################################################

This project is based on MarkovDecisionProcess (MDP) in Reinforcement Learning

I have leveraged the following repository/website:
https://gym.openai.com/

Analysis Pdf provided for introduction of problem, available data and RL solutioning.


**Frozen Lake:** 
A grid environment of a lake with holes, taken from Open AI Gym. Agent traverses a lake divided into 4x4 = 16 tiles. Each tile is either a frozen surface or a hole. Agent begins from the start tile ‘S’ and aims to reach the goal tile ‘G’. Agent earns rewards for avoiding obstacles i.e. holes and finding a path to the non-hole tile. However, frozen also indicates slippery tile and hence agent may not always go in the intended direction. There are only 4 possible directions for the agent to move – Up, Down, Right and Left. 
Forest Management: A non-grid environment with a growing forest, taken from hive mdptoolbox. Agent has a CUT or WAIT decision to make at every step. Depending on this decision, the transition states change between 0 and 1. When agent decides to CUT forest , it gets reward of +1 and forest transitions to state-0 whereas when agent decides to WAIT, forest passes to next state i.e. state-1 with probability ‘q’ and goes back to previous state i.e. state-0 with probability (1-q) because of forest fires. Agent makes a choice weather to risk going to final state and get higher reward or to receive +1 reward by cutting this forest at any given state. And in final state, if agent CUT forest, it earns reward r1 and goes back to state-0 in contrast to WAIT, where it can earn reward r2 and remains in current state with probability ‘q’ but goes to state-0 with probability (1-q).     
Part B)  Solving MDP1: Frozen Lake 8x8
current state with probability ‘q’ but goes to state-0 with probability (1-q).
epsilon (float) – Stopping criterion. The maximum change in the value function at each iteration is compared against epsilon. Once the change falls below this value, then the value function is considered to have converged to the optimal value function. 
https://pymdptoolbox.readthedocs.io/en/latest/api/mdp.html#mdptoolbox.mdp.PolicyIteration
This is a classic and simple grid-world wherein we have reward as +1 for the goal(final) state and 0 in case of obstacles(hole). Agent has to start from top-leftmost grid-cell  and find its way to the goal state by avoiding falling into holes. 


**Forest Management:**
A non-grid environment with a growing forest, taken from hive mdptoolbox. Agent has a CUT or WAIT decision to make at every step. Depending on this decision, the transition states change between 0 and 1. When agent decides to CUT forest , it gets reward of +1 and forest transitions to state-0 whereas when agent decides to WAIT, forest passes to next state i.e. state-1 with probability ‘q’ and goes back to previous state i.e. state-0 with probability (1-q) because of forest fires. Agent makes a choice weather to risk going to final state and get higher reward or to receive +1 reward by cutting this forest at any given state. And in final state, if agent CUT forest, it earns reward r1 and goes back to state-0 in contrast to WAIT, where it can earn reward r2 and remains in current state with probability ‘q’ but goes to state-0 with probability (1-q).    
