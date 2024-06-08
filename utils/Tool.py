import math
import time

import numpy as np
import torch
import random
import os


def set_random_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED']=str(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)
    else:
        torch.cuda.manual_seed(seed)

def getlocaltime():
    date = time.strftime('%y-%m-%d', time.localtime())
    current_time = time.strftime('%H:%M:%S', time.localtime())