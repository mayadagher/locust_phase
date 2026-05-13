'''_____________________________________________________IMPORTS____________________________________________________________'''

import numpy as np
import xarray as xr
from tqdm import tqdm

'''_____________________________________________________CLASSES____________________________________________________________'''

class Agent:
    def __init__(self, id:int, pos:np.ndarray[float], theta:float):
        self.id = id
        self.pos = pos
        self.theta = theta
        self.state = None

