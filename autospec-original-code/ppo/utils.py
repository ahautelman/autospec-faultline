class RolloutBuffer:
    def __init__(self, max_len = 200):
        self.actions = []
        self.states = []
        self.max_len = 200
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
    
    def __len__(self):
        return len(self.actions)