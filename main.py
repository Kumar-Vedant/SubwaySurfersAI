import os
from datetime import datetime

import torch

from agent import Agent
from customEnv import Game
from wrappers import apply_wrappers

# create path to store trained models
model_path = os.path.join("/home/kumar-vedant/Documents/Development/subwaySurfersAI/models", datetime.now().strftime("%d-%m-%Y-%H_%M_%S"))
os.makedirs(model_path, exist_ok=True)

NUM_EPISODES = 500
CKPT_SAVE_INTERVAL = 50
TRAINING = True

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

env = Game()
env = apply_wrappers(env)

if TRAINING:
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
else:
    # set exploration off (very low)
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n, epsilon=0.05, eps_min=0, eps_decay=0)
    # load the trained model
    ckpt_name = "06-12-2024-13_46_59/model_500.pt"
    agent.load_model(os.path.join("models", ckpt_name))

for episode in range(NUM_EPISODES):
    print("Episode:", episode)
    done = False
    state, _ = env.reset()
    total_reward = 0

    # play the episode until a terminal state is reached
    while not done:
        # pick an action at the state reached
        action = agent.choose_action(state)
        # take that action and record the experience
        new_state, reward, done, truncated, info  = env.step(action)
        total_reward += reward

        if TRAINING:
            # store the experience in the replay memory buffer and start a learning step
            agent.store_in_memory(state, action, reward, new_state, done)
            agent.learn()

        state = new_state
    
    # log the results of this episode
    print("Total reward:", total_reward, "Epsilon:", agent.epsilon, "Size of replay buffer:", len(agent.replay_buffer), "Learn step counter:", agent.learn_step_counter)

    if TRAINING and (episode+1)%CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(episode + 1) + ".pt"))

env.close()