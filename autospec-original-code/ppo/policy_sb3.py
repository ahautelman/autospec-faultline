import numpy as np
import gymnasium as gym
from tqdm import trange
import statistics
from stable_baselines3.common.callbacks import EvalCallback
import sys 


from ppo.utils import RolloutBuffer
from ppo.actor_critic import ActorCritic
from refinement.goal import Goal
# from refinement.graph import Node
from refinement.utils import CacheStates
import pickle
from stable_baselines3 import PPO

def sample_policy(env: gym.Env, observation:np.ndarray, policy:PPO, goal:Goal):
    
    final_terminated = False
    total_reward = 0
    traj = [observation]
    while True:
        action, _ = policy.predict(observation)
        # print(action)
        observation, reward, terminated, truncated, info = env.step(action)
        # env.render()
        traj.append(observation)
        total_reward+=reward

        final_terminated = info['is_success']
        

        if final_terminated or terminated or truncated:
            break
    
    return final_terminated, total_reward, info, traj


def train_policy(env: gym.Env, start_node, end_node, n_episodes=3000, minimum_reach=0.9):


    # env.set_abstract_states(start_node, end_node)
    env.set_abstract_states(start_node, end_node)
    eval_callback = EvalCallback(env, eval_freq=1000,
                             deterministic=True, render=True, )
    
    
    model = PPO("MlpPolicy", env, verbose=0, device = "cpu")
    model.learn(total_timesteps = n_episodes, progress_bar=True)
    model.save("ppo_cartpole")
    return model

def test_policy(policy: PPO, env: gym.Env, start_node, end_node, n_episodes=3000):
    
    env.set_abstract_states(start_node, end_node)
    
    cached_states = CacheStates()

    reach = []
    rewards = [0]
    episodes = trange(n_episodes, desc='reach')
    trajectories = []
    for episode in episodes:
        
        observation, _ = env.reset()
        # env.render()
        
        reached, reward, info, traj= sample_policy(env, observation, policy, end_node.goal)
        reach.append(reached)
        rewards.append(reward)
        trajectories.append((traj, reached))
        cached_states.insert(observation[:2], reached)

        
        episodes.set_description(f"Current reach: {sum(reach)/len(reach):.2f}, total_reach: {sum(reach)}, reward: {statistics.mean(rewards):.2f}±{statistics.stdev(rewards):.1f}")
        
    pickle.dump(trajectories, open(f"{start_node.name}_{end_node.name}_traj.pkl", "wb"))    
    pickle.dump(cached_states.return_dataset(), open(f"{end_node.name}_data.pkl", "wb"))

    return sum(reach)/len(reach), cached_states
    
