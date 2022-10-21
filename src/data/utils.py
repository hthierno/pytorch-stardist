import random
from collections import defaultdict
import time

import numpy as np
import torch
import tifffile


class TimeTracker:
    def __init__(self):
        self.dict_start_end = defaultdict( list )
        self.phases = defaultdict( list )
    
    def tic(self, phase:str):
        self.dict_start_end[phase].append( [time.time(), None] )
    
    def tac(self, phase:str):
        end=time.time()
        start = self.dict_start_end[phase][-1][0]
        self.dict_start_end[phase][-1][-1] = end
        self.phases[phase].append( end-start )
    
    def get_duration(self, phase):
        return self.phases[phase]
    def __getitem__(self, phase):
        return self.phases[phase]


def load_tif(path):
    with tifffile.TiffFile(path) as tif:
        return tif.asarray().astype("float32")

def save_tif(path, tif):
    tifffile.imsave(path, tif)


def seed_all(seed):
    """
    Utility function to set seed across all pytorch process for repeatable experiment
    """
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    """
    Utility function to set random seed for DataLoader
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)