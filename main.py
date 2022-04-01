import numpy as np
import os
import torch
a = [True, False, True, True]
b = [True, False, True, True]
z = torch.tensor(a)
z += torch.sum(a and b)
print(z)