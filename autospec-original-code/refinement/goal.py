import numpy as np

class AbstractState():

    def __init__(self, x: float, y: float, height: float, width: float):
        self.x = x
        self.y = y
        self.height = height
        self.width = width

    def sample_state(self):
        return np.array([np.random.uniform(low=self.x, high=self.x + self.width),
                         np.random.uniform(low=self.y, high=self.y + self.height)])
        
        

class Goal(AbstractState):

    def __init__(self, x:float, y:float, height:float, width:float):
        super().__init__(x, y, height, width)
        # self.current_goal = self.sample_state()

    def reset(self):
        self.current_goal = self.sample_state()
        return self.current_goal
        
    def predicate(self, state:np.ndarray):

        return self.current_goal[0] - 1.5 <= state[0] <= self.current_goal[0] + 1.5 \
                and self.current_goal[1] - 1.5 <= state[1] <= self.current_goal[1] + 1.5
    
    def in_goal_region(self, state:np.ndarray):

        return self.x <= state[0] <= self.x + self.width \
                and self.y <= state[1] <= self.y + self.height
    
    def reward(self, state:np.ndarray):
        if self.predicate(state[:2]):
            return 10
        else:
            
            return state[3] - np.sqrt(np.sum((state[:2] - self.current_goal)**2))
        

class ModifiedGoal(Goal):
    def __init__(self,  x:float, y:float, height:float, width:float, classifier, reachable:bool=False):
        super().__init__(x, y, height, width)
        self.classifier = classifier
        self.reachable = reachable

    def sample_state(self):
        state = super().sample_state()

        while not self.classifier.predict(state[:2].reshape(1,-1))[0] == self.reachable:
            state = super().sample_state()
        
        return state
