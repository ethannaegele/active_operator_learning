import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader
from omegaconf import DictConfig
#from hydra.utils import instantiate
#import wandb
from dataclasses import dataclass

from al4pde.prob_models.prob_model import ProbModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@dataclass
class ConvTapSpec:
    name: str
    #block_idx: int
    enabled: bool
    in_channels: int
    out_channels: int
    kernel_size: int = 3
    stride: int = 1
    padding: int = 1
    use_pool: bool = True
    pool_type: str = "max"
    adaptive_pool: bool = True
    adaptive_pool_output_size: int = 8
    pool_kernel_size: int = 2
    pool_stride: int = 2
    activation: nn.Module = nn.ReLU()


@dataclass
class FCTapSpec:
    name: str
    enabled: bool
    in_channels: int
    out_channels: int
    use_end_activation: bool = False
    activation: nn.Module = nn.ReLU()

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
                 conv_taps: list[ConvTapSpec], 
                 fc_taps: list[FCTapSpec],
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
                 conv_taps: list[ConvTapSpec], 
                 fc_taps: list[FCTapSpec],
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
    

def build_conv_heads_sequential_from_taps(conv_taps: list[ConvTapSpec], 
                                          input_channels: int = 3) -> nn.Sequential:
    layers = []
    for tap in conv_taps:
        if tap.use_pool:
            if tap.pool_type == "max":
                if not tap.adaptive_pool:
                    if tap.name == "last":
                        raise ValueError("Must use Adaptive pooling at last conv tap")
                    pool = nn.MaxPool1d(kernel_size=tap.pool_kernel_size,
                                stride=tap.pool_stride)
                else:
                    pool = nn.AdaptiveMaxPool1d(tap.adaptive_pool_output_size)
            elif tap.pool_type in {"average", "avg"}:
                if not tap.adaptive_pool:
                    if tap.name == "last":
                        raise ValueError("Must use Adaptive pooling at last conv tap")
                    pool = nn.AvgPool1d(kernel_size=tap.pool_kernel_size,
                                        stride=tap.pool_stride)
                else:
                    pool = nn.AdaptiveAvgPool1d(tap.adaptive_pool_output_size)
            else:
                raise ValueError(f"pool_type is not recognized. Expected" \
                "pool_type in {'max', 'avg', 'average'}. Got {tap.pool_type}")
        else:
            if tap.name == "last":
                raise ValueError("Must use Adaptive pooling at last conv tap,"
                                "but pooling is disabled")
            pool = nn.Identity() # if no pooling, do nothing at this step

        if tap.name == "first":
            in_channels = input_channels
        else:
            in_channels = tap.in_channels

        layers.append(
            nn.Conv1d(in_channels=in_channels,
                    out_channels=tap.out_channels,
                    kernel_size=tap.kernel_size,
                    stride=tap.stride,
                    padding=tap.padding))
        layers.append(tap.activation)
        layers.append(pool)

    return nn.Sequential(*layers)

def build_fc_heads_sequential_from_taps(fc_taps: list[FCTapSpec],
                                        input_channels: int = 128,
                                        output_channels: int = 1) -> nn.Sequential:
    layers = []
    for tap in fc_taps:
        if tap.name in {"last", "only"}:
            out_channels = output_channels
        else:
            out_channels = tap.out_channels

        if tap.name in {"first", "only"}:
            in_channels = input_channels
        else:
            in_channels = tap.in_channels
        layers.append(nn.Linear(in_features=in_channels,
                                        out_features=out_channels))
        if tap.name != "last" or tap.use_end_activation:
            layers.append(tap.activation)

    return nn.Sequential(*layers)


class UncertaintyModelFullTrajectory1dFlexible(ProbModel):
    def __init__(self,
                input_channels: int, # usually 3 for 1d problems
                hidden_units: int, # ignored here
                output_channels: int, # traj_len
                input_grid_dim: int,
                fully_connected_out_features: int,
                conv_taps: list[ConvTapSpec], 
                fc_taps: list[FCTapSpec], 
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
        super().__init__(task, 
                         training_type, 
                         t_train, 
                         batch_size, 
                         val_period, 
                         vis_period, 
                         loss)
        self.time_grid_independent = False
        # self.conv_heads = build_conv_heads_from_taps(conv_taps=conv_taps)
        # #self.conv_taps_by_block = {}
        # self.fc_heads = build_fc_heads_from_taps(fc_taps=fc_taps)
        # #self.fc_heads_by_block = {}
        self.distribution = distribution.strip().lower()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_heads_sequential = build_conv_heads_sequential_from_taps(conv_taps=conv_taps, 
                                                                           input_channels=self.input_channels)
        if len(self.conv_heads_sequential) == 0:
            raise RuntimeError("self.conv_heads_sequential has not been built" \
            "properly; has length 0")

        # last three layers are [nn.Conv1d, activation, pool]
        if not isinstance(self.conv_heads_sequential[-3], torch.nn.modules.conv.Conv1d):
            raise RuntimeError("Third to last layer of self.conv_heads_sequential"
            f"should be Conv1d, got {self.conv_heads_sequential[-3]}")

        if not isinstance(self.conv_heads_sequential[-1], 
                          (torch.nn.modules.pooling.AdaptiveMaxPool1d, 
                            torch.nn.modules.pooling.AdaptiveAvgPool1d)):
            raise RuntimeError("Last layer of self.conv_heads_sequential should be Adaptive" \
            f"pooling, got {self.conv_heads_sequential[-1]}")
        last_conv = self.conv_heads_sequential[-3]
        last_pool = self.conv_heads_sequential[-1]

        # flattening step: [B, last_conv.out_channels, last_pool.output_size] -> [B, last_conv.out_channels * last_pool.output_size]
        self.first_fc_input_channels = last_conv.out_channels * last_pool.output_size
        self.fc_heads_sequential = build_fc_heads_sequential_from_taps(fc_taps=fc_taps, 
                                                                       input_channels=self.first_fc_input_channels,
                                                                       output_channels=self.output_channels)
        if len(self.fc_heads_sequential) == 0:
            raise RuntimeError("self.fc_heads_sequential has not been built" \
            "properly; has length 0")
        assert self.distribution in ["normal"], "Distribution must be 'normal'. Other distributions not supported currently."

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

    def forward(self, xx, grid, pde_param=None, t_idx=None,
                time_coord: Tensor | None = None):
        x = self.build_conv_input(xx, grid, pde_param=pde_param, t_idx=t_idx)
        x = self.conv_heads_sequential(x) # [B, last_conv.out_channels, last_pool.output_size]
        flat = nn.Flatten() 
        x = flat(x) # [B, last_conv.out_channels * last_pool.output_size]
        x = self.fc_heads_sequential(x) # [B, self.output_channels]
        return x

    def roll_out(self, 
                xx: Tensor,
                grid: Tensor,
                final_step: int,
                pde_param: Tensor | None = None,
                t_idx: Tensor | None = None,
                time_coord: Tensor | None = None,): # [B, T], time coordinate built by the model wrapper
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

    def unc_roll_out(self, xx: Tensor,
                    grid: Tensor,
                    final_step: int,
                    pde_param: Tensor | None = None,
                    t_idx: Tensor | None = None,
                    return_features=False, ):
        raise NotImplementedError
        

        
