import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from race_car_env import RaceCarEnv
from dqn_agent_unique import DQNAgent, QNetwork # Use your unique agent
import os

def print_model_summary(model_path="partially_trained_model.pth"):
    """
    Loads the trained model and prints its architecture and parameter count.
    """
    print("="*30)
    print("🤖 Model Architecture Summary 🤖")
    print("="*30)
    
    # We need to instantiate the agent to get the model structure
    env = RaceCarEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # We use a dummy agent just to hold the network
    # We pass 'dummy' for paths because we don't need them
    agent = DQNAgent(state_size, action_size) 
    
    # Load the saved weights into our dummy model
    if os.path.exists(model_path):
        agent.q_network.load_state_dict(torch.load(model_path))
        print(f"Successfully loaded model weights from {model_path}\n")
    else:
        print(f"Warning: No model file found at {model_path}. Displaying untrained architecture.\n")

    # Use torchinfo to print a clean summary
    # (batch_size, state_size)
    summary(agent.q_network, input_size=(1, state_size))
    print("\n")


def plot_training_stats(stats_file="training_stats.json"):
    """
    Loads the JSON stats file and plots the training performance.
    """
    print("="*30)
    print("📈 Training Performance Plots 📈")
    print("="*30)
    
    if not os.path.exists(stats_file):
        print(f"Error: Stats file not found at {stats_file}")
        print("Please run main.py to train the model and generate the file.")
        return

    # Load the data
    with open(stats_file, 'r') as f:
        data = json.load(f)

    if not data:
        print("Error: Stats file is empty. No data to plot.")
        return

    # Extract data into lists
    episodes = [s['episode'] for s in data]
    rewards = [s['total_reward'] for s in data]
    steps = [s['steps'] for s in data]
    epsilon = [s['epsilon'] for s in data]
    laps = [s['laps'] for s in data]

    # --- 1. Print Summary Table ---
    print("📊 Key Training Statistics 📊")
    print(f"Total Episodes Trained: {len(episodes)}")
    print(f"Max Reward: {np.max(rewards):.2f}")
    print(f"Average Reward: {np.mean(rewards):.2f}")
    print(f"Average Steps: {np.mean(steps):.2f}")
    print(f"Total Laps Completed: {np.sum(laps)}")
    print("\n")

    # --- 2. Calculate Moving Average for Rewards ---
    # A moving average helps smooth out the noisy reward signal
    window_size = 50 
    if len(rewards) >= window_size:
        rewards_ma = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    else:
        rewards_ma = [] # Not enough data to plot moving average

    # --- 3. Create Plots ---
    
    # Plot 1: Reward per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, rewards, label='Total Reward', alpha=0.5)
    if len(rewards_ma) > 0:
        # Plot moving average offset by the window size
        plt.plot(episodes[window_size-1:], rewards_ma, label=f'{window_size}-Episode Moving Avg', color='red', linewidth=2)
    plt.title('Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_rewards.png') # Save the plot
    print("Saved 'plot_rewards.png'")

    # Plot 2: Steps per Episode
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, steps, label='Steps Taken', color='blue')
    plt.title('Steps Taken per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_steps.png') # Save the plot
    print("Saved 'plot_steps.png'")

    # Plot 3: Epsilon Decay
    plt.figure(figsize=(12, 6))
    plt.plot(episodes, epsilon, label='Epsilon (Exploration Rate)', color='green')
    plt.title('Epsilon Decay Over Time')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon Value')
    plt.legend()
    plt.grid(True)
    plt.savefig('plot_epsilon.png') # Save the plot
    print("Saved 'plot_epsilon.png'")

    print("\nAll plots generated. Displaying plots...")
    plt.show() # Show all plots at the end


if __name__ == "__main__":
    # Use the model you've been training
    model_to_analyze = "partially_trained_model.pth" 
    
    print_model_summary(model_path=model_to_analyze)
    plot_training_stats(stats_file="training_stats.json")