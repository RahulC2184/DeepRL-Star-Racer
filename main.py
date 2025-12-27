import pygame
import torch
import numpy as np
from race_car_env import RaceCarEnv
from dqn_agent import DQNAgent, QNetwork # Use your unique agent
import os
import json # <-- NEW: For saving stats

# --- NEW: Function to load/initialize stats ---
def load_stats(stats_file):
    if os.path.exists(stats_file):
        with open(stats_file, 'r') as f:
            return json.load(f)
    return []
# --- END NEW ---

def main():
    env = RaceCarEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    
    num_episodes = 1500
    
    model_path = "partially_trained_model.pth"
    optimizer_path = "optimizer_state.pth"
    buffer_path = "replay_buffer.pth"
    stats_file = "training_stats.json" # <-- NEW: Stats file name
    
    start_episode = 0
    
    # --- NEW: Load previous stats ---
    all_stats = load_stats(stats_file)
    if all_stats:
        # If we have stats, resume from the next episode
        start_episode = all_stats[-1]['episode'] + 1
    # --- END NEW ---

    if os.path.exists(model_path) and os.path.exists(optimizer_path) and os.path.exists(buffer_path):
        print("Found a partially trained model. Continuing training.")
        agent.q_network.load_state_dict(torch.load(model_path))
        agent.optimizer.load_state_dict(torch.load(optimizer_path))
        agent.replay_buffer.buffer = torch.load(buffer_path)
        
        # Adjust epsilon if resuming from stats
        if start_episode > 0:
            agent.epsilon = all_stats[-1]['epsilon']
            print(f"Resuming from episode {start_episode} with epsilon {agent.epsilon:.4f}")
        else:
            # Fallback if stats file is missing but model isn't
            # This will decay from start
            pass 

    success_lap_threshold = 5
    success_reward_threshold = 3000
    successful_episodes = 0

    for episode in range(start_episode, num_episodes):
        state, _ = env.reset()
        state = np.array(state)
        
        total_reward = 0
        steps_taken = 0
        done = False
        truncated = False
        
        env.current_episode = episode # Pass episode number to render

        while not done and not truncated:
            action = agent.choose_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = np.array(next_state)
            
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            steps_taken += 1
            
            agent.update()

            env.render()
            pygame.time.wait(10) # Small delay for visualization

        print(f"Episode: {episode}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.4f}, Steps: {steps_taken}, Laps: {info.get('laps', 0)}")

        # --- NEW: Save stats after each episode ---
        stats = {
            'episode': episode,
            'total_reward': total_reward,
            'steps': steps_taken,
            'epsilon': agent.epsilon,
            'laps': info.get('laps', 0)
        }
        all_stats.append(stats)
        
        with open(stats_file, 'w') as f:
            json.dump(all_stats, f, indent=4)
        # --- END NEW ---

        if info.get('laps', 0) > 0 and total_reward > success_reward_threshold:
            successful_episodes += 1
        else:
            successful_episodes = 0

        if successful_episodes >= success_lap_threshold:
            print(f"Agent reached success criteria for {success_lap_threshold} consecutive episodes. Training finished.")
            torch.save(agent.q_network.state_dict(), "trained_model.pth")
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(optimizer_path):
                os.remove(optimizer_path)
            if os.path.exists(buffer_path):
                os.remove(buffer_path)
            break

        if episode % 10 == 0:
            torch.save(agent.q_network.state_dict(), model_path)
            torch.save(agent.optimizer.state_dict(), optimizer_path)
            torch.save(agent.replay_buffer.buffer, buffer_path)

    # --- Final simulation loop (remains unchanged) ---
    print("Training finished. Running final simulation with learned policy.")
    agent.epsilon = 0.0
    state, _ = env.reset()
    state = np.array(state)
    done = False
    truncated = False
    
    while not done and not truncated:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        state = np.array(next_state)
        env.render()
        pygame.time.wait(30)

    env.close()

if __name__ == "__main__":
    main()