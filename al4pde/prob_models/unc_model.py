import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
#from hydra.utils import instantiate
#import wandb

from al4pde.prob_models.prob_model import ProbModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class UncertaintyModelFullTrajectory1d(ProbModel):
    """
    A model for predicting uncertainty in full trajectories.
    Must use same grid dimensions for training and inference.
    """
    def __init__(self, 
                 input_channels: int, # must match channel calculations depending on if pde_param is included
                 hidden_units: int, 
                 output_channels: int, # number of time steps, probably final_step or similar, or traj_len 
                 input_grid_dim: int,
                 fully_connected_out_features: int | None, # ignored
                 task,
                 training_type,
                 t_train, 
                 batch_size, 
                 val_period,
                 vis_period,
                 loss,
                 activation: nn.Module,
                 max_pool_output_size: int | None = None, # ignored
                 distribution: str = "normal"
                 ):
        super().__init__(task, training_type, t_train, batch_size, val_period, vis_period, loss)
        self.input_grid_dim = input_grid_dim
        self.activation = activation
        self.time_grid_independent = False
        self.output_channels = output_channels

        self.distribution = distribution.strip().lower()
        assert self.distribution in ["normal"], "Distribution must be 'normal'. Other distributions not supported currently."

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            self.activation,
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1),
            self.activation
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1),
            self.activation,
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * (input_grid_dim // 4),
                      out_features=256),
            self.activation,
            nn.Linear(in_features=256,
                      out_features=128),
            self.activation,
            nn.Linear(in_features=128,
                      out_features=128),
            self.activation,
            nn.Linear(in_features=128,
                      out_features=output_channels)

        )
    
    def build_conv_input(self, xx, grid, pde_param=None, t_idx=None):
        # xx: [B, X, initial_step, C]
        b, nx = xx.shape[0], xx.shape[1]

        # [B, X, initial_step, C] -> [B, initial_step*C, X]
        xx_features = torch.flatten(xx, start_dim=-2)      # [B, X, initial_step*C]
        xx_features = xx_features.permute(0, 2, 1)         # [B, initial_step*C, X]

        # grid: [X, 1] or [B, X, 1] -> [B, 1, X]
        if grid.dim() == 2:
            grid_features = grid.unsqueeze(0).expand(b, -1, -1)
        else:
            grid_features = grid
        grid_features = grid_features.permute(0, 2, 1)

        features = [xx_features, grid_features]

        if pde_param is not None:
            # [B, P] -> [B, P, X]
            param_features = pde_param[:, :, None].expand(-1, -1, nx)
            features.append(param_features)

        return torch.cat(features, dim=1)

    def forward(self, 
                xx: Tensor, 
                grid: Tensor | None,  # probably many of these arguments cannot be integrated
                pde_param: Tensor | None = None, 
                t_idx: Tensor | None = None,
                #num_time_steps: int = 41,
                time_coord: Tensor | None = None, # [B, T], time coordinate built by the model wrapper
                ):
        if xx.shape[1] != self.input_grid_dim:
            raise ValueError(
                f"Uncertainty model expected X={self.input_grid_dim}, "
                f"but got X={xx.shape[1]}. Make sure inputs are reduced "
                f"to the same resolution used during initialization."
            )
        x = self.build_conv_input(xx, grid, pde_param=pde_param, t_idx=t_idx) # preprocess step
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.fully_connected(x)
        return x
    
    def roll_out(self, 
                 xx: Tensor,
                 grid: Tensor, 
                 final_step: int, 
                 pde_param: Tensor | None = None,
                 t_idx: Tensor | None = None,
                 #num_time_steps: int = 41,
                 time_coord: Tensor | None = None, # [B, T], time coordinate built by the model wrapper
                 ):
        """Roll out is the same as a single forward pass since the
        uncertainty model is full trajectory.
        """
        return self.forward(xx,
                            grid = grid,
                            pde_param = pde_param,
                            t_idx = t_idx,
                            #num_time_steps=num_time_steps,
                            time_coord=time_coord,
                            )[..., :final_step]
    
    def unc_roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        raise NotImplementedError("""This model is uncertainty only.
                                   Uncertainties are provided via forward pass/roll_out of the
                                  UncertaintyModel or via the unc_roll_out method of the
                                  MeanUncModel.""")


class UncertaintyModelFullTrajectory1dTimeIndep(ProbModel):
    """
    A model for predicting uncertainty in full trajectories.
    Must use same grid dimensions for training and inference.
    """
    def __init__(self, 
                 input_channels: int, # must match channel calculations depending on if pde_param is included
                 hidden_units: int,
                 output_channels: int | None, # ignored
                 input_grid_dim: int | None, # ignored
                 fully_connected_out_features: int, 
                 task,
                 training_type, 
                 t_train, 
                 batch_size, 
                 val_period,
                 vis_period, 
                 loss,
                 activation: nn.Module,
                 max_pool_output_size: int = 8,
                 distribution: str = "normal"
                 ):
        super().__init__(task, training_type, t_train, batch_size, val_period, vis_period, loss)
        self.activation = activation
        self.time_grid_independent = True

        self.distribution = distribution.strip().lower()
        assert self.distribution in ["normal"], "Distribution must be 'normal'. Other distributions not supported currently."

        self.block_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_channels,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            self.activation,
            nn.MaxPool1d(kernel_size=2,
                         stride=2)
        )
        self.block_2 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1),
            self.activation
        )
        self.block_3 = nn.Sequential(
            nn.Conv1d(hidden_units, hidden_units, 3, padding=1),
            self.activation,
            nn.AdaptiveMaxPool1d(max_pool_output_size)
        )
        self.fully_connected = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * max_pool_output_size,
                      out_features=256),
            self.activation,
            nn.Linear(in_features=256,
                      out_features=128),
            # self.activation,
            # nn.Linear(in_features=128,
            #           out_features=128),
            self.activation,
            nn.Linear(in_features=128,
                      out_features=fully_connected_out_features) # [B, H=fully_connected_out_features]

        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=fully_connected_out_features + 1,
                      out_features=16),
            self.activation,
            nn.Linear(in_features=16,
                      out_features=8),
            self.activation,
            nn.Linear(in_features=8,
                      out_features=1)
        )
    
    
    def build_conv_input(self, xx, grid, pde_param=None, t_idx=None):
        # xx: [B, X, initial_step, C]
        b, nx = xx.shape[0], xx.shape[1]

        # [B, X, initial_step, C] -> [B, initial_step*C, X]
        xx_features = torch.flatten(xx, start_dim=-2)      # [B, X, initial_step*C]
        xx_features = xx_features.permute(0, 2, 1)         # [B, initial_step*C, X]

        # grid: [X, 1] or [B, X, 1] -> [B, 1, X]
        if grid.dim() == 2:
            grid_features = grid.unsqueeze(0).expand(b, -1, -1)
        else:
            grid_features = grid
        grid_features = grid_features.permute(0, 2, 1)

        features = [xx_features, grid_features]

        if pde_param is not None:
            # [B, P] -> [B, P, X]
            param_features = pde_param[:, :, None].expand(-1, -1, nx)
            features.append(param_features)

        return torch.cat(features, dim=1)

    def forward(self, 
                xx: Tensor, 
                grid: Tensor | None,  # probably many of these arguments cannot be integrated
                pde_param: Tensor | None = None, 
                t_idx: Tensor | None = None,
                #num_time_steps: int = 41,
                time_coord: Tensor | None = None, # [B, T], time coordinate built by the model wrapper
                ):
        if time_coord is None:
            raise ValueError("time_coord cannot be None for time grid independent models")
        num_time_steps = time_coord.shape[1]
        x = self.build_conv_input(xx, grid, pde_param=pde_param, t_idx=t_idx) # preprocess step
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x) # [B, hidden_units * 8]
        x = self.fully_connected(x) # [B, H]
        x = x[:, None, :].expand(-1, num_time_steps, -1) # [B, T, H]
        x = torch.cat([x, time_coord.unsqueeze(-1)], dim=-1)     # [B, T, H + 1]
        x = self.decoder(x) # [B, T, 1]
        x = x.squeeze(-1) # [B, T]
        return x
        # could also experiment with expanding across time after block3, then just use one block of fully connected layers?

    def roll_out(self, 
                     xx: Tensor,
                     grid: Tensor, 
                     final_step: int, 
                     pde_param: Tensor | None = None,
                     t_idx: Tensor | None = None,
                     #num_time_steps: int = 41,
                     time_coord: Tensor | None = None, # [B, T], time coordinate built by the model wrapper
                     ):
        """Roll out is the same as a single forward pass since the
        uncertainty model is full trajectory.
        """
        return self.forward(xx,
                            grid = grid,
                            pde_param = pde_param,
                            t_idx = t_idx,
                            #num_time_steps=num_time_steps,
                            time_coord=time_coord
                            )[..., :final_step]
        
    def unc_roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        raise NotImplementedError("""This model is uncertainty only.
                                    Uncertainties are provided via forward pass/roll_out of the
                                    UncertaintyModel or via the unc_roll_out method of the
                                    MeanUncModel.""")
      

