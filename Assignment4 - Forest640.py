from hiive.mdptoolbox.mdp import ValueIteration, PolicyIteration, QLearning, PolicyIterationModified
from hiive.mdptoolbox.example import forest
import gym
import numpy as np
#import sys
import math
from numpy.random import choice
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
np.random.seed(101)

P, R = forest(S=640, r1=100, r2= 10)  # To measure effect of changing number of states, please only change S parameter to 64 and the constant discount factor/gamma to 0.9

def Compute_Policy_Reward(P, R, policy, test_count=100, gamma=0.999):
    episodes = P.shape[-1] * test_count
    rewards_sum = 0
    for s in range(P.shape[-1]):
        s_reward = 0
        for s_episode in range(test_count):
            episode_reward = 0
            disc = 1
            while True:
                action = policy[s]
                probab = P[action][s]
                array = list(range(len(P[action][s])))
                next_s =  choice(array, 1, p=probab)[0]
                reward = R[s][action] * disc
                episode_reward += reward
                disc = disc*gamma
                if next_s == 0:
                    break
            s_reward += episode_reward
        rewards_sum += s_reward
    return rewards_sum / episodes

def Process_VI(P, R, epsilon, discount=0.999):
    vi_df = pd.DataFrame(columns=["Epsilon","Log10 Eps","Policy", "Iterations", 
                                  "Time", "Mean Reward", "Value Function"])
    for eps in epsilon:
        vi = ValueIteration(P, R, gamma=discount, epsilon=eps, max_iter=1000,skip_check=True)
        vi.run()
        avg_reward = Compute_Policy_Reward(P, R, vi.policy)
        data = [float(eps),math.log10((eps)) ,vi.policy, vi.iter, vi.time,avg_reward, vi.V]
        df_length = len(vi_df)
        vi_df.loc[df_length] = data
    print('My VI dataframe w.r.t Changing Epsilon \n',vi_df)
    
    plt.style.use('ggplot')  
    plt.ylabel('Convergence Time (s)')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('VI Stats - Epsilon vs Convergence Time (s)')
    plt.plot(list(vi_df["Log10 Eps"]),list(vi_df["Time"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

    plt.style.use('ggplot')  
    plt.ylabel('Convergence Iterations')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('VI Stats - Epsilon vs Iterations-To-Converge')
    plt.plot(list(vi_df["Log10 Eps"]),list(vi_df["Iterations"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

    plt.style.use('ggplot')
    plt.ylabel('Mean Rewards')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('VI Stats - Epsilon vs Reward')
    plt.plot(list(vi_df["Log10 Eps"]),list(vi_df["Mean Reward"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

def Process_PI(P, R, epsilon, discount=0.999):
    pi_df = pd.DataFrame(columns=["Epsilon","Log10 Eps","Policy", "Iterations", 
                                  "Time", "Mean Reward", "Value Function"])
    for eps in epsilon:
        pi = PolicyIterationModified(P, R, gamma=discount, epsilon=eps, max_iter=1000,skip_check=True)
        pi.run()
        #print(pi.run_stats)#['Error'])
        avg_reward = Compute_Policy_Reward(P, R, pi.policy)
        data = [float(eps),math.log10((eps)) ,pi.policy, pi.iter, pi.time,avg_reward, pi.V]
        df_length = len(pi_df)
        pi_df.loc[df_length] = data

    print('My PI dataframe w.r.t Changing Epsilon\n',pi_df)
    pi_policy = pi_df.Policy
    print(pi_policy[0])
    plt.style.use('ggplot')  
    plt.ylabel('Convergence Time (s)')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('PI Stats - Epsilon vs Convergence Time (s)')
    plt.plot(list(pi_df["Log10 Eps"]),list(pi_df["Time"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

    plt.style.use('ggplot')  
    plt.ylabel('Convergence Iterations')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('PI Stats - Epsilon vs Iterations-To-Converge')
    plt.plot(list(pi_df["Log10 Eps"]),list(pi_df["Iterations"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

    plt.style.use('ggplot')
    plt.ylabel('Mean Rewards')  
    plt.xlabel('Log10 - Epsilon')
    plt.title('PI Stats - Epsilon vs Reward')
    plt.plot(list(pi_df["Log10 Eps"]),list(pi_df["Mean Reward"]), 'x-')
    #plt.grid()
    #plt.legend(loc='best')
    plt.show()

def Process_QL(P, R, discount, alpha_dec,epsilon,epsilon_decay, n_iter):
    q_df = pd.DataFrame(columns=["Max Iterations","Iters Conv","Alpha Decay", 
                                 "Epsilon", "Epsilon Decay", "Mean Reward",
                                 "Time", "Policy", "Value Function",
                                 "Training Rewards"])
    count = 0
    for itr in n_iter:
        for eps in epsilon:
            for eps_dec in epsilon_decay:
                for a_dec in alpha_dec:
                    ql = QLearning(P, R, discount, alpha_decay=a_dec, epsilon=eps,epsilon_decay=eps_dec, n_iter=itr)
                    ql.run()
                    ql_reward = Compute_Policy_Reward(P, R, ql.policy)
                    count += 1
                    print("{}: {}".format(count, ql_reward))
                    #print(ql.policy) 
                    ql_stats = ql.run_stats
                    ql_rews = [s['Reward'] for s in ql_stats]
                    ql_info = [itr,ql. ,a_dec, eps, eps_dec,ql_reward, 
                                ql.time, ql.policy, ql.V, ql_rews]                      
                    q_df.loc[len(q_df)] = ql_info
    return q_df


if __name__ == "__main__":
    # Starting With Value Iteration
    Process_VI(P, R, epsilon=[0.1, 0.001, 0.0001, 0.0000001, 0.000000001])  # last 2 are e-1M and e-1B
    pi_policy=[]
    # Starting with Policy Iteration
    Process_PI(P, R, epsilon=[0.1, 0.001, 0.0001, 0.0000001, 0.000000001])  # last 2 are e-1M and e-1B   
    # Starting with Q-Learning
    alpha = 0.1 # This is the by default value and we are going to keep it thatw ay
    disc_factor = 0.999  # This is constant discount
    alpha_decs = [0.99, 0.999]
    eps = [1.0, 0.1,0.001]
    eps_dec = [0.99, 0.9]
    iters = [10000, 1000000]
    q_df = Process_QL(P, R, discount=disc_factor, alpha_dec=alpha_decs, 
                epsilon=eps, epsilon_decay=eps_dec, n_iter=iters)
    print('\nQ-Learning Dataframe\n',q_df)
    # Grouping my dataframe v=by different factors
    print('Performance w.r.t Iteration\n',q_df.groupby("Iterations").mean())
    print('Performance  w.r.t Epsilon\n',q_df.groupby("Epsilon").mean())
    print('Performance  w.r.t Epsilon Decay Rate\n',q_df.groupby("Epsilon Decay").mean())
    print('Performance  w.r.t Alpha Decay Rate\n',q_df.groupby("Alpha Decay").mean())

    

