import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import pickle

from collections import deque

class CacheStates:

    def __init__(self, maxlen:int = 10000):

        self.states = deque(maxlen=maxlen)
        self.reachable = deque(maxlen=maxlen)
    
    def __len__(self):
        return len(self.states)
    
    def insert(self, state:np.ndarray, reach:bool):

        self.states.append(state)
        self.reachable.append(reach)
    
    def return_dataset(self):
        return np.array(self.states), np.array(self.reachable)
    
    def clear(self):
        
        self.states.clear()
        self.reachable.clear()


def train_model(cached_states:CacheStates):

    model = LogisticRegression()
    model.fit(*cached_states.return_dataset())
    pickle.dump(model, open("svc.pkl", "wb"))
    pickle.dump(cached_states.return_dataset(), open("data.pkl", "wb"))
    return model

