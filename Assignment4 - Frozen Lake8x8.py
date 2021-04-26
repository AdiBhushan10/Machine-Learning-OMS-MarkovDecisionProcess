import warnings
warnings.filterwarnings('ignore')
import gym
import time
import math
import numpy as np
import pandas as pd
from gym.envs.toy_text.frozen_lake import FrozenLakeEnv
import matplotlib.pylab as plt
import seaborn as sns
import random
np.random.seed(100)

MAPS = {
    "4x4": [
        "SFFF",
        "FHFF",
        "FHFH",
        "HFFG"
    ],
    "8x8": [
        "SFFHFFFF",
        "FFFFFFHF",
        "FFFHFFFF",
        "FHFFFHFF",
        "FFFFFFFF",
        "FHFFFFHF",
        "FHFHHFHF",
        "FFFHFFFG"
    ],
    "15x15": [
        "SFFFFFFFFFFFFFF",
        "FFFFFFFFFFFHFFF",
        "FFFHFFFFFFFFFFF",
        "FFFFFHFFHFHFFFF",
        "FFFHFFFFFFFHFFF",
        "FHHFFFHFFFFFFFF",
        "FHFFHFHFFFFHFFF",
        "FFFFFFFFFFFFFFF",
        "FFFHFFFFFFFFFFH",
        "FFFFFHFFFFHFFFF",
        "FFFHFFFFFFFFHFF",
        "FHHFFFHFFFFFFFF",
        "FHFFHFHFFFFFFFF",
        "FFFFFFHFFFFHFFF",
        "FFFHFFFFFFHFFFG"
    ]
}

def Policy_testing(env, policy, n_epoch=1000):
    rewards = []
    episode_counts = []
    for i in range(n_epoch):
        current_state = env.reset()
        ep = 0
        done = False
        episode_reward = 0
        while not done and ep < 1000:
            ep += 1
            act = int(policy[current_state])
            new_state, reward, done, _ = env.step(act)
            episode_reward += reward
            current_state = new_state
        rewards.append(episode_reward)
        episode_counts.append(ep)
    mean_reward = sum(rewards)/len(rewards)
    mean_eps = sum(episode_counts)/len(episode_counts)
    return mean_reward, mean_eps, rewards, episode_counts

def Value_Iteration(env, discount, epsilon):
    start = time.time()
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    policy = np.zeros((1, number_of_states))
    value_list = np.zeros((1, number_of_states))
    old_value_list = value_list.copy()
    episode = 0
    max_change = 1
    sigma = discount
    while max_change > epsilon:
        episode += 1
        for s in range(number_of_states):
            assigned_value = -np.inf
            for a in range(number_of_actions):        
                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = old_value_list[0][new_state]
                    cand_value = 0
                    if done:
                        cand_value = reward 
                    else:
                        cand_value = reward + sigma*value_new_state
                    total_cand_value += cand_value*prob 
                        
                if total_cand_value > assigned_value:
                    assigned_value = total_cand_value
                    policy[0][s] = a
                    value_list[0][s] = assigned_value
        changes = np.abs(value_list - old_value_list)
        max_change = np.max(changes)
        old_value_list = value_list.copy()
    end = time.time()
    duration = end-start
    print("For discount factor: {} and epsilon: {}, Convergence occurs at Episode: {} in Duration: {} seconds".format(discount,epsilon,episode, duration))
    return policy[0], episode, duration


def Policy_Iteration(env, discount=0.9, epsilon=1e-3):
    start = time.time()
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    policy = np.random.randint(number_of_actions, size=(1,number_of_states))
    value_list = np.zeros((1, number_of_states))
    episode = 0
    sigma = discount
    policy_stable = False
    while not policy_stable:
        episode += 1
        eval_acc = True
        while eval_acc:
            eps = 0
            for s in range(number_of_states):
                v = value_list[0][s] 
                a = policy[0][s]
                total_val_new_state = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]
                    cand_value = 0
                    if done:
                        cand_value = reward                     
                    else:
                        cand_value = reward + sigma*value_new_state
                    total_val_new_state += cand_value*prob 
                value_list[0][s] = total_val_new_state
                eps = max(eps, np.abs(v-value_list[0][s]))
            if eps < epsilon:
                eval_acc = False

        policy_stable = True
        for s in range(number_of_states):
            old_action = policy[0][s]
            max_value = -np.inf
            for a in range(number_of_actions):
                total_cand_value = 0
                for prob, new_state, reward, done in env.P[s][a]:
                    value_new_state = value_list[0][new_state]
                    cand_value = 0
                    if done:
                        cand_value = reward
                    else:
                        cand_value = reward + sigma*value_new_state
                    total_cand_value += prob*cand_value
                if total_cand_value > max_value:
                    max_value = total_cand_value
                    policy[0][s] = a
            if old_action != policy[0][s]:
                policy_stable = False
    end = time.time()
    duration = end-start
    print("For discount factor: {} and epsilon: {}, Convergence occurs at Episode: {} in Duration: {} seconds".format(discount,epsilon,episode, duration))
    return policy[0], episode, duration

def PI_VI_modeling(env, discount, epsilon):
    vi_dict = {}
    print("Value Iteration (VI) - Process Begins...")
    for dis in discount:
        vi_dict[dis] = {}
        for eps in epsilon:
            vi_dict[dis][eps] = {}
            vi_policy, vi_solve_iter, vi_solve_time = Value_Iteration(env, dis, eps)
            vi_mrews, vi_meps, _, __ = Policy_testing(env, vi_policy)    
            vi_dict[dis][eps]["mean_reward"] = vi_mrews
            vi_dict[dis][eps]["mean_eps"] = vi_meps
            vi_dict[dis][eps]["iteration"] = vi_solve_iter
            vi_dict[dis][eps]["duration"] = vi_solve_time
            vi_dict[dis][eps]["policy"] = vi_policy

    print("Policy Iteration (PI) - Process Begins...")
    pi_dict = {}
    for dis in discount:
        pi_dict[dis] = {}
        for eps in epsilon:
            pi_dict[dis][eps] = {}

            pi_policy, pi_solve_iter, pi_solve_time = Policy_Iteration(env, dis, eps)
            pi_mrews, pi_meps, _, __ = Policy_testing(env, pi_policy)    
            pi_dict[dis][eps]["mean_reward"] = pi_mrews
            pi_dict[dis][eps]["mean_eps"] = pi_meps
            pi_dict[dis][eps]["iteration"] = pi_solve_iter
            pi_dict[dis][eps]["duration"] = pi_solve_time
            pi_dict[dis][eps]["policy"] = pi_policy
    return vi_dict, pi_dict

def map_discretize(the_map):
    size = len(the_map)
    dis_map = np.zeros((size,size))
    for i, row in enumerate(the_map):
        for j, loc in enumerate(row):
            if loc == "S":
                dis_map[i, j] = 0
            elif loc == "F":
                dis_map[i, j] = 0
            elif loc == "H":
                dis_map[i, j] = -1
            elif loc == "G":
                dis_map[i, j] = 1
    return dis_map

def reshape_policy(policy):
    size = int(np.sqrt(len(policy)))
    pol = np.asarray(policy)
    pol = pol.reshape((size, size))
    return pol

def policy_visual(map_size, policy):
    map_name = str(map_size)+"x"+str(map_size)
    data = map_discretize(MAPS[map_name])
    np_pol = reshape_policy(policy)
    plt.imshow(data, interpolation="nearest", cmap="Accent")

    for i in range(np_pol[0].size):
        for j in range(np_pol[0].size):
            arrow = '\u2190'
            if np_pol[i, j] == 1:
                arrow = '\u2193'
            elif np_pol[i, j] == 2:
                arrow = '\u2192'
            elif np_pol[i, j] == 3:
                arrow = '\u2191'
            text = plt.text(j, i, arrow,
                           ha="center", va="center", color="k")
    plt.show()

def graph_visual(dictionary, value, size, variable, main_title, discount=True):
    value = "Mean {}".format(value)
    the_df = pd.DataFrame(columns=[variable, value])
    for k, v in dictionary.items():
        for val in v:
            if discount:   # This boolean value (True) represent - If we are plotting w.r.t dicount factor
                dic = {variable: k, value: float(val)}
            else:
                dic = {variable: math.log10(k), value: float(val)} # If we plot w.r.t log, then I took log base 10               
            the_df = the_df.append(dic, ignore_index=True)
    plt.figure()
    plt.style.use('ggplot')
    sns.lineplot(x=variable, y=value, legend="full",ci=None, data=the_df).set(title=main_title)
    plt.show()

def create_mydict(the_dict):
    print('dict is:\n',the_dict)
    discount_rewards = {}
    discount_iterations = {}
    discount_times = {}
    for disc in the_dict:
        discount_rewards[disc] = []    
        discount_iterations[disc] = []    
        discount_times[disc] = []
        for eps in the_dict[disc]:
            discount_rewards[disc].append(the_dict[disc][eps]['mean_reward'])
            discount_iterations[disc].append(the_dict[disc][eps]['iteration'])        
            discount_times[disc].append(the_dict[disc][eps]['duration'])          
    epsilon_rewards = {}
    epsilon_iterations = {}
    epsilon_times = {}
    for eps in the_dict[0.999]:
        epsilon_rewards[eps] = []    
        epsilon_iterations[eps] = []    
        epsilon_times[eps] = []
        for disc in vi_dict:
            epsilon_rewards[eps].append(the_dict[disc][eps]['mean_reward'])
            epsilon_iterations[eps].append(the_dict[disc][eps]['iteration'])        
            epsilon_times[eps].append(the_dict[disc][eps]['duration'])          
    return discount_rewards, discount_iterations, discount_times, epsilon_rewards, epsilon_iterations, epsilon_times

def Q_learning(env, discount, total_episodes, alpha=0.1, decay_rate=None,
               min_epsilon=0.01):
    start = time.time()
    number_of_states = env.observation_space.n
    number_of_actions = env.action_space.n
    qtable = np.zeros((number_of_states, number_of_actions))
    learning_rate = alpha
    gamma = discount
    epsilon = 1.0
    max_epsilon = 10.0 
    min_epsilon = 0.01
    if not decay_rate:
        decay_rate = 1./total_episodes
    rewards = []
    for episode in range(int(total_episodes)):
        # environment reset required
        state = env.reset()
        step = 0
        done = False
        total_reward = 0
        while True:
            exp_exp_tradeoff = random.uniform(0,1)
            if exp_exp_tradeoff > epsilon:
                b = qtable[state, :]
                action = np.random.choice(np.where(b == b.max())[0])
            else:
                action = env.action_space.sample()   
            new_state, reward, done, info = env.step(action)
            total_reward += reward
            if not done:
                qtable[state, action] = qtable[state, action] + learning_rate*(reward + gamma*np.max(qtable[new_state, :]) - qtable[state, action])
            else:
                qtable[state, action] = qtable[state,action] + learning_rate*(reward - qtable[state,action])
            state = new_state
            if done:
                break 
        rewards.append(total_reward)
        epsilon = max(max_epsilon -  decay_rate * episode, min_epsilon)     
    end = time.time() 
    duration = end-start
    print("For discount factor: {} and total epsilodes: {}, convergence occurs in Duration: {} seconds".format(discount,episode, duration))
    return np.argmax(qtable, axis=1), total_episodes, duration, qtable, rewards

def QL_modeling(env, discount, total_episodes, alphas, decay_rates, mute=False):
    
    min_epsilon = 0.01
    
    q_dict = {}
    for dis in discount:
        q_dict[dis] = {}
        for eps in total_episodes:
            q_dict[dis][eps] = {}
            for alpha in alphas:
                q_dict[dis][eps][alpha] = {}
                for dr in decay_rates:
                    q_dict[dis][eps][alpha][dr] = {}
                    q_policy, q_solve_iter, q_solve_time, q_table, rewards = Q_learning(env, dis, eps, alpha, dr, min_epsilon)
                    q_mrews, q_meps, _, __ = Policy_testing(env, q_policy)
                    q_dict[dis][eps][alpha][dr]["mean_reward"] = q_mrews
                    q_dict[dis][eps][alpha][dr]["mean_eps"] = q_meps
                    q_dict[dis][eps][alpha][dr]["q-table"] = q_table
                    q_dict[dis][eps][alpha][dr]["rewards"] = rewards 
                    q_dict[dis][eps][alpha][dr]["iteration"] = q_solve_iter
                    q_dict[dis][eps][alpha][dr]["time_spent"] = q_solve_time
                    q_dict[dis][eps][alpha][dr]["policy"] = q_policy
                    print("Mean reward: {} - at mean episodes: {}".format(q_mrews, q_meps))
                    print("Iteration: {} - in duraion: {}".format(q_solve_iter, q_solve_time))

    return q_dict

def create_dataframe(the_dict):
    the_df = pd.DataFrame(columns=["Discount Rate", "Training Episodes", "Learning Rate", 
                                   "Decay Rate", "Reward", "Time Spent", "Mean Episodes","Convergence Iterations"])
    for dis in the_dict:
        for eps in the_dict[dis]:
            for lr in the_dict[dis][eps]:
                for dr in the_dict[dis][eps][lr]:
                    rew = the_dict[dis][eps][lr][dr]["mean_reward"]
                    time_spent = the_dict[dis][eps][lr][dr]["time_spent"] #.total_seconds()
                    mean_epis = the_dict[dis][eps][lr][dr]["mean_eps"]
                    iters_con = the_dict[dis][eps][lr][dr]["iteration"]
                    dic = {"Discount Rate": dis, "Training Episodes": eps, "Learning Rate":lr, 
                           "Decay Rate":dr, "Reward": rew, "Time Spent": time_spent, "Mean Episodes":mean_epis,"Convergence Iterations":iters_con}
                    the_df = the_df.append(dic, ignore_index=True)
    return the_df


if __name__ == "__main__":
    
    #env = gym.make("FrozenLake8x8-v0")
    env = FrozenLakeEnv(desc=MAPS["8x8"]) 
    #env = gym.make("FrozenLake-v0")
    #env = FrozenLakeEnv(desc=MAPS["15x15"]) 

    vi_dict, pi_dict = PI_VI_modeling(env, discount=[0.9, 0.99, 0.999, 1.0], 
											epsilon=[0.1, 0.001, 0.00001, 0.0000001])

    policy_visual(8, vi_dict[0.999][0.00001]['policy'])  #Replace 8 by 4 OR 15 - to study behavior for reduced OR increased number of states
    
    policy_visual(8, pi_dict[0.999][0.001]['policy'])  #Replace 8 by 4 OR 15 - to study behavior for reduced OR increased number of states
    
    sz=8 #4 OR 15 - to study behavior for reduced OR increased number of states
    VI_data = create_mydict(vi_dict)
    graph_visual(VI_data[0], value="Reward", size=sz,variable="Discount Factor", main_title='VI - Mean Reward w.r.t Discount Factor')  # avg rewards vs gamma
    graph_visual(VI_data[1], value="Iteration", size=sz, variable="Discount Factor",main_title='VI - Iterations w.r.t Discount Factor') # iter vs gamma
    graph_visual(VI_data[2], value="Time", size=sz, variable="Discount Factor",main_title='VI - Convergence Time w.r.t Discount Factor') # time vs gamma
    graph_visual(VI_data[3], value="Reward", size=sz, variable="Log10 Epsilon Value", main_title='VI - Mean Reward w.r.t Log10 Epsilon', discount=False) # rewards vs log epsilon
    graph_visual(VI_data[4], value="Iteration", size=sz, variable="Log10 Epsilon Value",main_title='VI - Iterations w.r.t Log10 Epsilon', discount=False) # iters vs log epsilon
    graph_visual(VI_data[5], value="Time", size=sz, variable="Log10 Epsilon Value",main_title='VI - Convergence Time w.r.t Log10 Epsilon',  discount=False) # time vs log epsilon
    PI_data = create_mydict(pi_dict)
    graph_visual(PI_data[0], value="Reward", size=sz, variable="Discount Factor",main_title='PI - Mean Reward w.r.t Discount Factor')
    graph_visual(PI_data[1], value="Iteration", size=sz, variable="Discount Factor",main_title='PI - Iterations w.r.t Discount Factor')
    graph_visual(PI_data[2], value="Time", size=sz, variable="Discount Factor",main_title='PI - Convergence Time w.r.t Discount Factor')
    graph_visual(PI_data[3], value="Reward", size=sz, variable="Log10 Epsilon Value", main_title='PI - Mean Reward w.r.t Log10 Epsilon', discount=False)
    graph_visual(PI_data[4], value="Iteration", size=sz, variable="Log10 Epsilon Value", main_title='PI - Iterations w.r.t Log10 Epsilon', discount=False)
    graph_visual(PI_data[5], value="Time", size=sz, variable="Log10 Epsilon Value", main_title='PI - Convergence Time w.r.t Log10 Epsilon', discount=False)
    
  
    ###########################################################
   
    print("Q-Learning (QL) - Process Begins...")
    env = gym.make("FrozenLake8x8-v0")
    q_dict = QL_modeling(env, discount= [0.999], total_episodes=[10000,5000,1000],
                            alphas=[0.1, 0.01], decay_rates=[1e-3])
    
    QL_data = create_dataframe(q_dict)
    print('QL Stats: \n', QL_data)

    print('Optimum Policy - Q-Learning')
    pol = q_dict[0.999][10000][0.1][1e-03]['policy']
    policy_visual(8, pol)
    #print('Optimum Policy with 5000 training episodes')
    #pol = q_dict[0.999][5000][0.1][1e-03]['policy']
    #policy_visual(8, pol)
    #print('Optimum Policy with 1000 training episodes')
    #pol = q_dict[0.999][1000][0.1][1e-03]['policy']
    #policy_visual(8, pol)

    plt.figure()
    plt.style.use('ggplot')
    plt.title('QL - Mean Reward (w/ learning rate) w.r.t total episodes ')
    sns.lineplot(x="Training Episodes", y="Reward", data=QL_data)
    plt.show()
    plt.title('QL - Mean Reward w.r.t Steps to Converge')
    sns.lineplot(x="Mean Episodes", y="Reward", data=QL_data)
    plt.show()
    plt.title('QL - Convergence Time w.r.t Steps to Converge')
    sns.lineplot(x="Time Spent", y="Reward", data=QL_data)
    plt.show()
    plt.title('QL - Steps to Converge w.r.t Time to Converge')
    sns.lineplot(y="Time Spent", x="Mean Episodes", data=QL_data)
    plt.show()

    

