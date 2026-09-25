import random
import numpy as np
import torch

def set_seed():
    random.seed(6304)
    np.random.seed(6304)
    torch.manual_seed(6304)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(6304)
        torch.cuda.manual_seed_all(6304)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False