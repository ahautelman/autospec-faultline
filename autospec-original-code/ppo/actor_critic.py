import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, continuous = True):
        super(Actor, self).__init__()

        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        self.continuous = continuous
        if continuous:
            self.actor_std = nn.Parameter(torch.ones(size = (action_dim, )))
    
    def forward(self, x):
        action_logits = self.actor(x)
        if self.continuous:
            dist = MultivariateNormal(loc = torch.tanh(action_logits), covariance_matrix=torch.diag(self.actor_std))
        else:
            dist = Categorical(logits=action_logits)
        
        return dist

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.critic(x)

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, continuous = True):
        super(ActorCritic, self).__init__()

        self.actor = Actor(state_dim, action_dim, continuous = True)
        self.critic = Critic(state_dim, action_dim)

        self.continuous = continuous

    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        dist = self.actor(state)

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        dist = self.actor(state)
        state_values = self.critic(state)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, state_values, dist_entropy
