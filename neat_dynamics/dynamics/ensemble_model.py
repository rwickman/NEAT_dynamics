from dynamics_model import DynamicsModel
from replay_memory import ReplayMemory

class EnsembleModel:
    def __init__(self, args, ac_dim, ob_dim):
        self.args = args
        
        self.dyn_models = []
        for _ in range(self.args.ensemble_size):
            self.dyn_models.append(DynamicsModel(self.args, ac_dim, ob_dim))

        self.replay_memory = ReplayMemory(self.args) 

    def train(self):
        states, actions, rewards, next_states = self.sample()
        num_data = states.shape[0]
        num_data_per_model = int(num_data / self.args.ensemble_size)
        total_loss = 0
        for i in range(self.args.ensemble_size):
            start_idx = i * num_data_per_model
            end_idx = (i + 1) * num_data_per_model 
            total_loss += self.dyn_models[i].update(
                states[start_idx:end_idx],
                actions[start_idx:end_idx],
                rewards[start_idx:end_idx],
                next_states[start_idx:end_idx])
        
        return (total_loss / self.args.ensemble_size).item()

    def add_to_replay(self, exps):
        self.replay_memory.append(exps)
    
    def sample(self, batch_size: int):
        samples = self.replay_memory.sample(batch_size * self.args.ensemble_size)
        states, actions, rewards, next_states = [], [], [], []
        for i in range(len(samples)):
            states.append(samples[i].state)
            actions.append(samples[i].action)
            rewards.append(samples[i].reward)
            next_states.append(samples[i].next_state)

        return states, actions, rewards, next_states

        