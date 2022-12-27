import torch
import random
import numpy as np


class ReplayMemory:
    def __init__(self, args):
        self._args = args
        self._memory = []
        # Pointer to end of memory
        self._cur_pos = 0
    
    def append(self, e_t):
        """Append experience."""
        if len(self._memory) >= self._args.memory_capacity:
            self._memory[self._cur_pos] = e_t
        else:
            self._memory.append(e_t)
        
        # Update end of memory
        self._cur_pos = (self._cur_pos + 1) %  self._args.memory_capacity 
    
    def add_rollouts(self, exps):
        for exp in exps:
            self.append(exp)

    def sample(self, batch_size):
        """Sample batch size experience replay."""
        return np.random.choice(self._memory, size=batch_size, replace=False)

    def current_capacity(self):
        return len(self._memory)