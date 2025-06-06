import sys
import os
from datetime import datetime

import matplotlib.pyplot as plt

import torch

from agent import Agent

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from customEnv import Game
from wrappers import apply_wrappers

# create path to store trained models
model_path = os.path.join("/home/kumar-vedant/Documents/Development/subwaySurfersAI/models", datetime.now().strftime("%d-%m-%Y-%H_%M_%S"))
os.makedirs(model_path, exist_ok=True)

NUM_EPISODES = 50000
CKPT_SAVE_INTERVAL = 100
TRAINING = True

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

env = Game()
env = apply_wrappers(env)

if TRAINING:
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    agent.load_model("/home/kumar-vedant/Documents/Development/subwaySurfersAI/PPO/classification_model_0.pt")

else:
    # set exploration off (very low)
    agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)
    # load the trained model
    ckpt_name = "26-12-2024-02_20_02/model_2500.pt"
    # agent.load_model(os.path.join("/home/kumar-vedant/Documents/Development/subwaySurfersAI/models", ckpt_name))
    # agent.load_model("/home/kumar-vedant/Documents/Development/subwaySurfersAI/classification_model_0.pt")
    agent.load_model("/home/kumar-vedant/Documents/Development/subwaySurfersAI/models/29-05-2025-01_44_32/model_1400.pt")


# enable interactive mode of matplotlib
plt.ion()

# initialize plot
fig, ax = plt.subplots()
rewards = []
episodes = []

line, = ax.plot([], [], 'b-')

ax.set_xlabel('Episodes')
ax.set_ylabel('Reward')
ax.set_title('Reward vs Episodes (Training Progress)')
ax.grid(True)


for episode in range(NUM_EPISODES):
    print("Episode:", episode)
    done = False
    state, _ = env.reset()
    total_reward = 0

    # play the episode until a terminal state is reached
    while not done:
        # pick an action at the state reached
        action, prob, value = agent.choose_action(state)
        # take that action and record the experience
        new_state, reward, done, truncated, info  = env.step(action)
        total_reward += reward

        if TRAINING:
            # store the experience in the replay memory buffer and start a learning step
            agent.store_in_memory(state, action, reward, prob, value, done)
            agent.learn(new_state)

        state = new_state
    
    # log the results of this episode
    print("Total reward:", total_reward, "Size of observations buffer:", len(agent.observations[0]), "Learn step counter:", agent.learn_step_counter)

    # record reward of this episode for plot
    rewards.append(total_reward)
    episodes.append(episode)

    # update plot
    line.set_xdata(episodes)
    line.set_ydata(rewards)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.001)

    if TRAINING and (episode+1)%CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(episode + 1) + ".pt"))

env.close()

plt.ioff()
plt.show()