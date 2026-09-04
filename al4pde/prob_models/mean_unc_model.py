import os
import time
import torch
from timeit import default_timer
from collections import OrderedDict
import wandb
import torch.nn as nn
import numpy as np
import hydra.utils
import functools

from al4pde.prob_models.prob_model import ProbModel
from al4pde.models.build_wrapper import build_wrapper
from al4pde.evaluation.analysis import batch_errors
from al4pde.utils import subsample_grid
from al4pde.evaluation.stats import LossUncCorr, UncAvg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MeanUncModel(ProbModel):
    def __init__(self, 
                 mean_model, # eg fno full traj
                 unc_model_factory, # cnn style model implemented in unc_model.py
                 noise,
                 noise_mode,
                 block_grad,
                 max_grad_norm,
                 step_batch,
                 optimizer_factory,
                 scheduler_factory,
                 name,
                 if_conditional,
                 *args,
                 time_coord_mode = "abs_time",
                 time_coord_norm_mode = None,
                 default_unc_traj_len = None,
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.unc_model_factory = unc_model_factory
        self.mean_model = mean_model
        self.unc_model = None
        self.noise = noise
        self.noise_mode = noise_mode
        self.block_grad = block_grad
        self.max_grad_norm = max_grad_norm
        self.step_batch = step_batch
        self.norm_max_counter = 0
        self.max_norm_stat = 0
        self.max_norm = 0
        assert noise_mode in ["constant", "channel_std_rel"]
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        #self.first_iter_nb = self.mean_model.first_iter_nb
        self.name = name
        self.if_conditional = if_conditional
        if time_coord_mode not in {"abs_time", "rel_time"}:
            raise ValueError("Invalid value of time_coord_mode:", time_coord_mode)
        self.time_coord_mode = time_coord_mode
        #self.time_coord_norm_mode = time_coord_norm_mode
        self.default_unc_traj_len = default_unc_traj_len
        self.stats.append(UncAvg("unc"))
        self.stats.append(LossUncCorr("corr_unc_loss", self.loss))

    def _build_time_coord(self, grid, t_idx, traj_len):
        """Method within class to build time coordinate.
        Placed within class to remove dependency on FNO full trajectory as
        the mean model."""
        bsize = grid.shape[0]
        rel_t = torch.arange(traj_len, device=grid.device, dtype=grid.dtype)

        if self.time_coord_mode == "abs_time":
            if t_idx is None:
                stored_t_idx = rel_t[None, :].repeat(bsize, 1)
            else:
                stored_t_idx = t_idx.to(device=grid.device, dtype=grid.dtype)[:, None] + rel_t[None, :]
            # Datasets expose t_idx in the loaded/reduced trajectory. Convert it
            # back to the solver-step index before asking the simulator for time.
            solver_t_idx = (stored_t_idx + getattr(self, "skip_initial_steps", 0)) * getattr(self, "reduced_resolution_t", 1)
            if hasattr(self.task, "sim") and hasattr(self.task.sim, "get_time"):
                time_coord = self.task.sim.get_time(solver_t_idx)
            else:
                time_coord = solver_t_idx
        else: # rel_time
            rel_t = rel_t[None, :].repeat(bsize, 1)
            solver_rel_t = rel_t * getattr(self, "reduced_resolution_t", 1)
            if hasattr(self.task, "sim") and hasattr(self.task.sim, "dt"):
                time_coord = solver_rel_t * self.task.sim.dt
            else:
                raise RuntimeError("Unable to find sim or sim.dt attributes" \
                "associated with task")
        return time_coord

    def _get_traj_len(self, final_step=None):
        if hasattr(self.unc_model, "output_channels"): # time grid dependent unc model has a time dim length built-in
            traj_len = self.unc_model.output_channels
        elif hasattr(self.mean_model, "traj_len"): # full trajectory mean model wrapper
            traj_len = self.mean_model.traj_len
        elif final_step is not None: # autoregressive models; only final_step is given, not traj_len
            traj_len = final_step
        elif self.default_unc_traj_len is not None: # autoregressive mean model, method is called without final_step
            traj_len = self.default_unc_traj_len
        else:
            raise RuntimeError("Could not determine the value of traj_len;" \
            "no wrapper traj_len, final_step, or default_unc_traj_len was provided. Cannot" \
            "determine how many time slices to provide uncertainties for.")
        return traj_len

    
    def _task_norm_helper(self, xx, grid, pde_param):
        """Safely obtain the normalized xx, grid, and pde_param.
        Temporary normalization steps until task norm is integrated
        with unc models."""
        self.task_norm = getattr(self.mean_model, "task_norm", None)
        if self.task_norm is not None:
            xx_norm = self.task_norm.norm_traj(xx)
            grid_norm = self.task_norm.norm_grid(grid)
            pde_param_norm = self.task_norm.norm_param(pde_param)
        else:
            xx_norm, grid_norm, pde_param_norm = xx, grid, pde_param
        return xx_norm, grid_norm, pde_param_norm

    def uncertainty(self, xx, grid, pde_param=None, t_idx=None):
        "Output shape: [B], different from uncertainty method for ProbModel"
        xx_unc, grid_unc, pde_param_unc = self._task_norm_helper(xx, grid, pde_param)
        if not self.if_conditional:
            pde_param_unc = None
        if getattr(self.unc_model, "time_grid_independent", False):
            traj_len = self._get_traj_len()
            time_coord = self._build_time_coord(grid, t_idx, traj_len)
        else:
            time_coord = None

        uncs_by_timestep = torch.exp(self.unc_model(xx_unc, 
                                                    grid_unc, 
                                                    pde_param=pde_param_unc, 
                                                    t_idx=t_idx,
                                                    time_coord=time_coord))
        return uncs_by_timestep.mean(dim=1) # scalar uncertainty for the entire trajectory
    
    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        return self.mean_model(xx, grid, pde_param=pde_param, t_idx=t_idx, return_features=return_features)
    
       
    
    def unc_roll_out(self, xx, grid, final_step, 
                     pde_param=None, t_idx=None, return_features=False,
                     #num_time_steps=41, 
                     time_coord=None):
        
        # right now, we assume self.mean_model is FNOFullTrajectory, 
        # which is initialized with traj_len, from which final_step is probably obtained
        if return_features:
            raise NotImplementedError("Returning features for MeanUncModel is not supported yet")
        mean_traj = self.mean_model.roll_out(xx, grid, final_step, pde_param, t_idx, False)
        # unc_traj needs to be the uncertainty vector expanded to match the trajectory

        # TEMP NORM STEPS UNTIL TASK NORM INTEGRATED WITH UNC MODEL

        xx_unc, grid_unc, pde_param_unc = self._task_norm_helper(xx, grid, pde_param)
        if not self.if_conditional:
            pde_param_unc = None

        if time_coord is None and getattr(self.unc_model, "time_grid_independent", False):
            traj_len = self._get_traj_len(final_step=final_step)
            time_coord = self._build_time_coord(grid, t_idx, traj_len)

        log_uncs_by_timestep = self.unc_model.roll_out(xx_unc, 
                                                       grid_unc, 
                                                       final_step, 
                                                       pde_param=pde_param_unc, 
                                                       t_idx=t_idx,
                                                       #num_time_steps=num_time_steps,
                                                       time_coord=time_coord
                                                       ) # log sigmas

        uncs_by_timestep = torch.exp(log_uncs_by_timestep) # sigmas 
        # spatial_shape: tuple, e.g. (nx1, nx2) or (nx1, nx2, nx3)
        # nc: number of channels
        spatial_shape = grid.shape[1:-1]
        nc = mean_traj.shape[-1] # num channels

        spatial_shape = tuple(spatial_shape)
        if uncs_by_timestep.shape[-1] < final_step:
            raise ValueError(f"Uncertainty trajectory is shorter than final_step: {uncs_by_timestep.shape[-1]} < {final_step}")
        
        if mean_traj.shape[-2] > final_step:
            mean_traj = mean_traj[..., :final_step, :]
        elif mean_traj.shape[-2] < final_step:
            raise ValueError(f"Mean trajectory is shorter than final_step: {mean_traj.shape[-2]} < {final_step}")
        uncs_by_timestep = uncs_by_timestep[..., :final_step]
        mask = torch.ones_like(uncs_by_timestep)
        mask[:, 0:self.initial_step] = 0.0
        uncs_by_timestep = uncs_by_timestep * mask # zero out initial uncs in autograd-friendly way
        #uncs_by_timestep[:, 0:self.initial_step] = 0.0 
        B, T = uncs_by_timestep.shape
        unc_traj = uncs_by_timestep.reshape(B, *([1] * len(spatial_shape)), T, 1)
        # expansion along channels is probably dubious for nc > 1 unless unc model is expanded to calculate unc for each channel
        unc_traj = unc_traj.expand(B, *spatial_shape, final_step, nc)
        return mean_traj, unc_traj

    def expected_model_change(self, xx, grid, final_step, pde_param, t_idx, num_mc_samples, use_nll_loss):
        pred, unc = self.unc_roll_out(xx, grid, final_step, pde_param, t_idx, time_coord=None)
        spatial_dim = self.task.spatial_dim
        reduction_slice = (slice(None),) + (0,) * spatial_dim + (slice(None),) + (0,) # slice like unc[:, 0, ..., 0, :, 0]
        log_sigma_t = torch.log(unc[reduction_slice].clamp_min(1e-8)) # [B, spatial, T, C] -> [B, T], assumes unc same for each channel
        
        score = 0
        for m in range(num_mc_samples):
            simulated_error = torch.normal(mean = torch.zeros_like(pred), std = unc)
            mc_yy = (pred + simulated_error).detach() # y = pred + mean zero error, so estimates actual yy
            # actual grad calculation step to estimate model change:
            if use_nll_loss:
                loss = self.gaussian_nll(pred, mc_yy, log_sigma_t, self.initial_step, reduction='mean')
                params = [p for p in list(self.mean_model.model.parameters()) + list(self.unc_model.parameters()) if p.requires_grad]
            else:
                params = [p for p in list(self.mean_model.model.parameters()) if p.requires_grad]
                loss = self.loss(pred, mc_yy, initial_step=self.initial_step)
            grads = torch.autograd.grad(loss, 
                                        params,
                                        retain_graph=m < num_mc_samples - 1,
                                        )
            grad_norm = torch.sqrt(sum((g.detach() ** 2).sum() for g in grads))
            score += grad_norm
        return score / num_mc_samples # MC estimate of expected model change, shape [1]
        

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        return self.mean_model.roll_out(xx, grid, final_step, pde_param, t_idx, return_features)
   
    def sample_trajectory(self, xx, grid, final_step, pde_param=None, t_idx=None):
        # placeholder behavior for sample trajectory
        return self.mean_model.roll_out(xx, grid, final_step, pde_param, t_idx, False)
    
    @property
    def val_loader(self):
        return self.mean_model.val_loader

    @property
    def train_loader(self):
        return self.mean_model.train_loader

    @property
    def train_loader_full_traj(self):
        return self.mean_model.train_loader_full_traj

    def reduce_input_res(self, xx, grid):
        return self.mean_model.reduce_input_res(xx, grid)
    
    def eval_pred(self, xx, yy, grid, pde_param=None, t_idx=None):
        xx = xx.to(device)
        grid = grid.to(device)
        yy = yy.to(device)
        pde_param = pde_param.to(device)
        t_idx = t_idx.to(device)
        time_coord = self._build_time_coord(grid, t_idx, yy.shape[-2])
        pred, unc = self.unc_roll_out(xx, 
                                      grid, 
                                      yy.shape[-2], 
                                      pde_param=pde_param, 
                                      t_idx=t_idx,
                                      return_features=False,
                                      time_coord=time_coord)
        onestep_pred = self.mean_model.one_step_pred(yy, grid, pde_param, t_idx)
        return {"pred": pred, 
                "unc": unc, 
                "onestep_pred": onestep_pred, 
                "yy": yy, 
                "param": pde_param,
                "grid": grid,
                "t_idx": t_idx}


    def validate(self, step, val_loader=None, train_loader=None, prefix=""):
        super().validate(step, 
                        val_loader=val_loader, 
                        train_loader=train_loader, 
                        prefix=prefix)
    
    def evaluate(self, step, loader, prefix, time_step_name = "total_epoch", extra_log = None):
        return super().evaluate(step, loader, prefix, time_step_name=time_step_name, extra_log=extra_log)

    def visualize(self, total_epoch):
        return self.mean_model.visualize(total_epoch)

    def gaussian_nll(self, mean_traj, yy, log_sigma_t, initial_step, reduction='mean'):
        if mean_traj.shape != yy.shape:
            raise ValueError(f"""mean_traj shape does not match yy: 
                             mean_traj.shape = {mean_traj.shape},
                             yy.shape = {yy.shape}""")
        if log_sigma_t.shape[-1] < yy.shape[-2]:
            raise ValueError(f"""log_sigma_t does not have enough uncertainty values to match yy:
                             log_sigma_t.shape[-1] = {log_sigma_t.shape[-1]} 
                             yy.shape[-2] = {yy.shape[-2]}""")
        spatial_shape = mean_traj.shape[1:-2]
        nc = mean_traj.shape[-1] # num channels
        spatial_shape = tuple(spatial_shape)
        B, T = log_sigma_t.shape
        log_sigma_t = log_sigma_t.reshape(B, *([1] * len(spatial_shape)), T, 1)
        log_sigma_t = log_sigma_t.expand(B, *spatial_shape, T, nc) # expand uncertainty to same shape as mean_traj and yy
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=mean_traj.device, dtype=mean_traj.dtype))
        loss = 0.5 * (mean_traj - yy)**2 \
            * torch.exp(-2.0 * log_sigma_t) + \
                0.5 * log_2pi + log_sigma_t
        loss = loss[..., initial_step:, :]
        if reduction == 'mean':
            loss = loss.mean()
        else:
            raise ValueError("Only mean reduction is currently supported for gaussian_nll")
        return loss

    def init_training(self, al_iter, load_train_data=True, 
                      warm_start_model_states: dict[str, OrderedDict] | None = None,
                      #warm_start_unc_model_state: OrderedDict | None = None
                      ):
        if warm_start_model_states is not None:
            warm_start_mean_model_state = warm_start_model_states['mean']
        else:
            warm_start_mean_model_state = None
        self.mean_model.init_training(al_iter, 
                                      load_train_data=load_train_data, 
                                      warm_start_model_state=warm_start_mean_model_state)
        self.unc_model = self.unc_model_factory().to(device)

        if warm_start_model_states is not None and warm_start_model_states['unc'] is not None:
            self.unc_model.load_state_dict(warm_start_model_states['unc'])
            
        self.unc_model.train()

        total_unc_model_params = sum(p.numel() for p in self.unc_model.parameters() if p.requires_grad)
        print(f'Total uncertainty model parameters = {total_unc_model_params}')
        self.task_norm = self.mean_model.task_norm
        self.loss.init_training(self.task_norm, self.initial_step)
        self.optimizer = self.optimizer_factory(list(self.mean_model.model.parameters()) + list(self.unc_model.parameters()))
        self.scheduler = self.scheduler_factory(self.optimizer)
        self.first_iter_nb = self.mean_model.first_iter_nb

    
    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        t1 = default_timer()

        self.mean_model.train()
        self.unc_model.train()
        self.task_norm = getattr(self.mean_model, "task_norm", None)
        #self.unc_model.task_norm = self.mean_model.task_norm # not working yet - needs to be integrated into unc_model
        train_mse_sum = 0
        train_loss_sum = 0
        n = 0
        train_max_l2 = 0
        max_grad = -1
        grad_sum = 0
        grad_sum_n = 0
        norm_sum = 0
        norm_n = 0

        if self.noise_mode == "channel_std_rel" and self.task_norm is not None:
            std = self.noise * self.task_norm.std_channels
        elif self.noise_mode == "constant" or self.task_norm is None:
            std = self.noise
        else:
            raise ValueError(f"Invalid noise mode {self.noise_mode}")

        for batch_idx, (xx, yy, grid, pde_param, t_idx) in enumerate(self.train_loader):
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)
            t_idx = t_idx.to(device)
            _batch = yy.size(0)

            xx = xx + torch.normal(torch.zeros_like(xx), std) # noise behavior
            if self.block_grad:
                xx = xx.detach() 

            pred = self.mean_model.roll_out(xx, grid, yy.shape[-2], pde_param, t_idx)

            # currently, since uncertainty model does not normalize internally,
            # must normalize here
            xx_unc, grid_unc, pde_param_unc = self._task_norm_helper(xx, grid, pde_param)
            if not self.if_conditional:
                pde_param_unc = None
            if getattr(self.unc_model, "time_grid_independent", False):
                traj_len = self._get_traj_len(yy.shape[-2])
                time_coord = self._build_time_coord(grid, 
                                                  t_idx, 
                                                  traj_len)
            else:
                time_coord = None
            log_sigma_t = self.unc_model.roll_out(xx_unc, grid_unc, 
                                                  yy.shape[-2], 
                                                  pde_param=pde_param_unc, 
                                                  t_idx=t_idx,
                                                  #num_time_steps=yy.shape[-2],
                                                  time_coord=time_coord
                                                  )
            loss = self.gaussian_nll(pred, yy, log_sigma_t, self.initial_step, reduction='mean')

            self.optimizer.zero_grad()

            loss.backward()
            total_norm = 0
            model_parameters = list(self.mean_model.model.parameters()) + list(self.unc_model.parameters())

            for param in model_parameters:
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    grad_sum += grad.abs().mean().detach().cpu()
                    grad_sum_n += 1
                    max_grad = max(max_grad, grad.abs().max().detach().cpu().item())
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            norm_sum += total_norm
            norm_n += 1
            tb_idx = (batch_idx + current_epoch * len(self.train_loader))

            self.max_norm_stat = max(total_norm, self.max_norm_stat)
            if self.max_grad_norm is not None:
                if self.max_grad_norm < total_norm:
                    self.norm_max_counter += 1
                if tb_idx == 0:
                    self.max_grad_norm = 5 * total_norm
                elif current_epoch < 5:
                    self.max_grad_norm = 5 * max(total_norm, self.max_grad_norm / 5)  # add some slack in beginning
                else:
                    norm_update = min(total_norm, self.max_grad_norm)
                    self.max_grad_norm = 5 * (self.max_grad_norm / 5 * 0.95 + norm_update * 0.05)
                torch.nn.utils.clip_grad_norm_(model_parameters, self.max_grad_norm)

            self.optimizer.step()
            all_se = batch_errors(pred, yy, self.initial_step)
            train_mse_sum += all_se.mean().detach() * _batch
            train_max_l2 = max(torch.sqrt(all_se).max().detach().item(), train_max_l2)
            train_loss_sum += loss.detach().item() * _batch
            n += _batch

            if self.step_batch != -1 and tb_idx % self.step_batch == 0 and tb_idx < self.first_iter_nb * num_epoch:
                self.scheduler.step()

        train_rmse = torch.sqrt(train_mse_sum / n).item()
        train_avg_loss = train_loss_sum / n
        t2 = default_timer()
        avg_grad = grad_sum / grad_sum_n
        if self.step_batch == -1:
            self.scheduler.step()

        print("epoch:", str(current_epoch), "time:", np.round(t2 - t1, 2),
              "train_rmse:", np.round(train_rmse, 4), "train_loss", np.round(train_avg_loss, 4),
              "avg_grad", np.round(avg_grad.item(), 4), "max_grad", np.round(max_grad, 3),
              "grad_norm", np.round(norm_sum/ norm_n, 4), "max_grad_norm", self.max_grad_norm)

        if current_epoch % self.val_period == 0:
            wandb.log({
                self.name + "/train_rmse": train_rmse,
                self.name + "/train_avg_loss": train_avg_loss,
                self.name + "/train_max_l2": train_max_l2,
                self.name + "/lr": self.scheduler.get_lr()[0],
                self.name + "/max_grad": max_grad,
                self.name + "/grad_norm": norm_sum/ norm_n,
                self.name + "/norm_cap_counter": self.norm_max_counter,
                self.name + "/norm_cap_rate": np.round(self.norm_max_counter / (self.val_period * len(self.train_loader)), 4),
                self.name + "/norm_max": self.max_norm_stat,
                self.name + "/avg_grad": avg_grad,
                "total_epoch": total_epoch,
            },
            )
            self.norm_max_counter = 0
            self.max_norm_stat = 0
    
    
    def train_n_epoch(self, 
                      al_iter, 
                      num_epoch, 
                      step_offset, 
                      vis=True, 
                      prefix="",
                      is_last=False,
                      warm_start_model_states: dict[str, OrderedDict] | None = None,
                      #warm_start_unc_model_state: OrderedDict | None = None
                      ):
        total_time = 0
        self.init_training(al_iter, 
                           load_train_data=True,
                           warm_start_model_states=warm_start_model_states,
                           #warm_start_unc_model_state=warm_start_unc_model_state
                           )
        for i in range(num_epoch):
            t = time.time()
            self.train_single_epoch(i, step_offset + i, num_epoch)
            total_time += time.time() - t
            if i % self.val_period == 0:
                self.validate(step_offset + i, prefix=prefix)
            if vis:
                if (i > 0 and i % self.vis_period == 0) or i == num_epoch - 1:
                    self.visualize(step_offset + i)
        return total_time
        
        
    
    def get_extra_state(self):
        return {"optimizer": self.optimizer.state_dict(), "scheduler": self.scheduler.state_dict()}

    def set_extra_state(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

def build_mean_unc_model(task, cfg):
    if task.spatial_dim != 1:
        raise NotImplementedError("Only 1D spatial tasks supported currently.")
    name = cfg.get("name", "mean_unc_model")
    mean_model = build_wrapper(task, cfg.model_wrapper, name=name + "/mean_model")
    grid = subsample_grid(task.get_grid(1), task.reduced_resolution)
    input_grid_dim = grid.shape[1] # for 1D tasks
    grid_channels = grid.shape[-1]
    input_channels = task.initial_step * task.num_channels + \
          grid_channels
    if cfg.unc_model.get("if_conditional", True):
        input_channels += task.num_pde_param # add the pde parameter(s) as channel(s) under if_conditional
    if "traj_len" in cfg.unc_model:
        output_channels = cfg.unc_model.traj_len # autoregressive fno support
    elif "traj_len" in cfg.model_wrapper.model:
        output_channels = cfg.model_wrapper.model.traj_len # full trajectory fno support
    else:
        output_channels = None
    unc_model_class = hydra.utils.get_class(cfg.unc_model._target_)
    unc_model_factory = functools.partial(unc_model_class, 
                                          input_channels = input_channels,
                                          hidden_units = cfg.unc_model.hidden_units,
                                          output_channels = output_channels,
                                          input_grid_dim = input_grid_dim,
                                          fully_connected_out_features = cfg.unc_model.get("fully_connected_out_features", 128),
                                          task = task,
                                           training_type = cfg.model_wrapper.training_type,
                                           t_train = cfg.model_wrapper.t_train,
                                           batch_size = cfg.model_wrapper.batch_size,
                                           val_period = cfg.model_wrapper.val_period,
                                           vis_period = cfg.model_wrapper.vis_period,
                                           loss = mean_model.loss,
                                           activation = hydra.utils.instantiate(cfg.unc_model.activation),
                                           max_pool_output_size = cfg.unc_model.get("max_pool_output_size", 8),
                                           distribution = cfg.unc_model.distribution
                                           
                                             )
    return MeanUncModel(mean_model,
                        unc_model_factory,
                        cfg.model_wrapper.noise,
                        cfg.model_wrapper.noise_mode,
                        cfg.model_wrapper.block_grad,
                        cfg.model_wrapper.max_grad_norm,
                        cfg.model_wrapper.step_batch,
                        hydra.utils.instantiate(cfg.model_wrapper.optimizer,
                                                 _partial_=True),
                        hydra.utils.instantiate(cfg.model_wrapper.scheduler, 
                                                _partial_=True,),
                        name,
                        cfg.unc_model.if_conditional,
                        task,
                        cfg.model_wrapper.training_type,
                        cfg.model_wrapper.t_train,
                        batch_size = cfg.model_wrapper.batch_size,
                        val_period = cfg.model_wrapper.val_period,
                        vis_period = cfg.model_wrapper.vis_period,
                        loss = mean_model.loss,
                        time_coord_mode=cfg.unc_model.get("time_coord_mode", "abs_time"),
                       # time_coord_norm_mode=cfg.unc_model.time_coord_norm_mode
                        default_unc_traj_len=cfg.unc_model.get("traj_len", None)
                    )





    
    


        
                 
