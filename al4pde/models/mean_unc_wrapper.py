import torch
import numpy as np
from timeit import default_timer
from al4pde.models.wrapper import ModelWrapper
import wandb
from collections import OrderedDict

from al4pde.models.pdebench_loss import get_loss
from al4pde.evaluation.analysis import batch_errors
from al4pde.models.model import Model
from al4pde.models.dataloader import NPYDataset
from al4pde.models.normalization import TaskNormalizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MeanUncertaintyWrapper(ModelWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    #def train_single_epoch(self, current_epoch, total_epoch, num_epoch):


