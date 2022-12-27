import torch
import torch.nn as nn
from torch import optim
from neat.novelty.dynamics.config import *

class DynamicsModel(nn.Module):
    """Dynamics model that predicts observations and rewards."""
    def __init__(self, args, ac_dim, ob_dim):
        self.args = args

        delta_layers = [nn.Linear(ob_dim + ac_dim, self.args.hidden_size), nn.ReLU()]
        for i in range(self.args.num_hidden):
            delta_layers.append(nn.Linear(self.args.hidden_size, self.args.hidden_size))
            delta_layers.append(nn.ReLU())
        #delta_layers.append(nn.Linear(self.args.hidden_size, ob_dim + 1))
        self.delta_trunk = nn.Sequential(*delta_layers)

        self.delta_head = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_size, ob_dim))


        self.reward_head = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_size, 1))
            

        self.optimizer = optim.Adam(
            self.delta_net.parameters(),
            self.args.lr)
        
        self.loss_fn = nn.MSELoss()
    
    def forward(self, states, actions):
        cat_inp = torch.cat([states, actions], dim=1)

        # Get the predictions
        latents = self.delta_net(cat_inp)
        pred_delta = self.delta_head(latents)
        pred_reward = self.reward_head(latents)
        
        next_state_preds = pred_delta + states
        return next_state_preds, pred_delta, pred_reward

    def predict(self, states, actions):
        next_state_preds, _, pred_reward = self(states, actions)
        return next_state_preds, pred_reward

    def update(self, states, actions, next_states, rewards):
        # Set the target to the difference of the states
        tgts = next_states - states

        # Get the delta predictions
        _, pred_delta, pred_reward = self(states, actions)

        loss = self.loss_fn(pred_delta, tgts)
        loss += self.loss_fn(pred_reward, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
        
        


