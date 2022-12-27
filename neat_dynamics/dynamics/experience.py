import torch
from dataclasses import dataclass
 
@dataclass
class Experience:
    state: torch.Tensor
    action: int
    reward: float
    next_state: torch.Tensor = None
