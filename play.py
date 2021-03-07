from unityagents import UnityEnvironment
import numpy as np
import torch
from collections import deque
import os as os
import sys
import config as config
from agent import DPG

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    hyper_params = config.Configuration()
    hyper_params.process_CLI(sys.argv[1:])

    env = UnityEnvironment(file_name=hyper_params.reacher_location)

    hyper_params.process_env(env)
    num_agents = hyper_params.num_agents
    action_size = hyper_params.action_size
    brain_name = hyper_params.brain_name
    n_episodes = hyper_params.n_episodes


    # create DPG Actor/Critic Agent
    agent = DPG(hyper_params)

    if(not hyper_params.model_dir): 
        hyper_params.model_dir = './trained_model/'
    # load trained agent
    agent.load_models(hyper_params.model_dir)

    scores = []                                 # list containing scores from each episode
    agent_scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    frames = 0

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=False)[brain_name] # reset the environment
        states = env_info.vector_observations             # get the current state of each agent
        agent_scores = np.zeros(num_agents)
        frames = 0
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]          # send all actions to tne environment
            next_states = env_info.vector_observations        # get next state (for each agent)
            rewards = env_info.rewards                        # get reward (for each agent)
            dones = env_info.local_done                       # see if episode finished
            agent_scores += rewards                           # update the score (for each agent)
            states = next_states                              # roll over states to next time step
            frames = frames+1
            if frames % 10 == 0:
                print('\rEpisode {}\t Frame: {:4}/1000 \t Score: {:.2f}'.format(i_episode, frames, np.mean(agent_scores)), end="")
            if np.any(dones):                                 # exit loop if episode finished
                break
        scores.append(np.mean(agent_scores))              # store episodes mean reward over agents
        print('\rEpisodes: {}\tLast score: {:.2f}\tAverage Score: {:.2f}\t\t'.format(i_episode, scores[-1], np.mean(scores)))
        

    env.close()
if __name__ == '__main__':
    main()