from PIL import ImageFile

import numpy as np
import os
import torch


def init_env(seed):
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    set_all_seeds(seed)

def set_all_seeds(seed):
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True