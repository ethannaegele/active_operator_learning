import time
import torch
from timeit import default_timer
from collections import OrderedDict
import wandb
import numpy as np
import hydra.utils

from torch import Tensor
import math

from al4pde.prob_models.prob_model import ProbModel
from al4pde.models.build_wrapper import build_wrapper
from al4pde.evaluation.analysis import batch_errors
from al4pde.models.fno_wrapper import FNOWrapper, FNOFullTrajectoryWrapper
from al4pde.models.arena_wrapper import ArenaWrapper
from al4pde.evaluation.stats import LossUncCorr, UncAvg


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sync_time(do_profile: bool) -> float | None:
    if do_profile:
        torch.cuda.synchronize()
        return time.time()

class MeanUnc(ProbModel):
    def __init__(self, 
                 model_wrapper: FNOWrapper | FNOFullTrajectoryWrapper | ArenaWrapper,
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
                 **kwargs
                 ):
        super().__init__(*args, **kwargs)
        self.model_wrapper = model_wrapper
        if isinstance(model_wrapper, ArenaWrapper):
            raise NotImplementedError("Internal MeanUnc behavior not yet " \
            "supported for PDEArena models.")
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
        self.stats.append(UncAvg("unc"))
        self.stats.append(LossUncCorr("corr_unc_loss", self.loss))


    def uncertainty(self, xx, grid, pde_param=None, t_idx=None):
        "Output shape: [B, X, 1, C]"
        mean, log_sigma = self.model_wrapper.forward_mean_unc(xx, 
                                                   grid, 
                                                   pde_param=pde_param,
                                                   t_idx=t_idx)
        spatial_ones = [1] * (mean.ndim - 3)
        if getattr(self.model_wrapper, "predicts_full_trajectory", False):
            B, T, C = log_sigma.shape
            # expand log_sigma from shape [B, T, C] to shape of mean
            log_sigma_expanded = log_sigma.reshape(B, *spatial_ones, T, C).expand_as(mean)
        else:
            # expand log_sigma from shape [B, C] to shape of mean
            B, C = log_sigma.shape
            log_sigma_expanded = log_sigma.reshape(B, *spatial_ones, 1, C).expand_as(mean)
        return mean, log_sigma_expanded
        
    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        return self.model_wrapper(xx, grid, pde_param=pde_param, 
                                  t_idx=t_idx, return_features=return_features)
    
       
    
    def unc_roll_out_log_sigma(self, xx, grid, final_step, 
                     pde_param=None, t_idx=None, return_features=False):
        """Unc roll out for Gaussian NLL training loop. Gives log sigma as
        uncertainty output.
        """
        
        unc_cell = []
        if self.training_type in {'autoregressive', 
                                  'teacher_forcing',
                                  'teacher_forcing_schedule',
                                  'full_teacher_forcing'}:
            pred = xx
            unc_cell = [torch.zeros_like(xx)]
            for t in range(self.initial_step, final_step):
                m, unc = self.uncertainty(xx, 
                                          grid, 
                                          pde_param=pde_param, 
                                          t_idx=t_idx,
                                          )
                pred = torch.cat((pred, m), -2)
                xx = torch.cat((xx[..., 1:, :], m), dim=-2)
                unc_cell.append(unc)
                if t_idx is not None:
                    t_idx += 1
            return pred, torch.concat(unc_cell, dim=-2)

        elif self.training_type == 'full_trajectory':
            m, unc = self.uncertainty(xx, 
                                    grid, 
                                    pde_param=pde_param, 
                                    t_idx=t_idx,
                                    )
            # up to initial step does not matter; it is discarded for NLL, and
            # is set to 0 for unc_roll_out so does not affect acquisition
            return m[..., 0:final_step, :], unc[..., 0:final_step, :]

        else:
            raise ValueError(self.training_type)

    def unc_roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        """Returns prediction and sigma, rather than log sigma. Used because
        acquisition requires nonnegative uncertainties.
        """
        pred, log_sigma = self.unc_roll_out_log_sigma(xx, 
                                                      grid, 
                                                      final_step, 
                                                      pde_param=pde_param, 
                                                      t_idx=t_idx, 
                                                      return_features=return_features)
        sigma = torch.exp(log_sigma)
        sigma[..., 0:self.initial_step, :] = 0.0
        return pred, sigma

        

    def roll_out(self, xx, grid, final_step, 
                 pde_param=None, t_idx=None, return_features=False):
        return self.model_wrapper.roll_out(xx, 
                                           grid, 
                                           final_step, 
                                           pde_param, 
                                           t_idx, 
                                           return_features)
   
    def sample_trajectory(self, xx, grid, final_step,
                          pde_param=None, t_idx=None):
        # placeholder behavior for sample trajectory
        return self.model_wrapper.roll_out(xx, 
                                           grid, 
                                           final_step, 
                                           pde_param, 
                                           t_idx, 
                                           False)
    
    @property
    def val_loader(self):
        return self.model_wrapper.val_loader

    @property
    def train_loader(self):
        return self.model_wrapper.train_loader

    @property
    def train_loader_full_traj(self):
        return self.model_wrapper.train_loader_full_traj

    def reduce_input_res(self, xx, grid):
        return self.model_wrapper.reduce_input_res(xx, grid)
    
    def eval_pred(self, xx, yy, grid, pde_param=None, t_idx=None):
        xx = xx.to(device)
        grid = grid.to(device)
        yy = yy.to(device)
        param = pde_param.to(device) if pde_param is not None else None
        t_idx = t_idx.to(device) if t_idx is not None else None
        pred, unc = self.unc_roll_out(xx, grid, yy.shape[-2], param, t_idx)
        if getattr(self.model_wrapper, "predicts_full_trajectory", False):
            onestep_pred = pred
        else:
            onestep_pred = self.model_wrapper.one_step_pred(yy, grid, param, 
                                                            t_idx)
        return {"pred": pred, 
                "onestep_pred": onestep_pred, 
                "yy": yy, 
                "param": param, 
                "unc": unc}
        

    def validate(self, step, val_loader=None, train_loader=None, prefix=""):
        super().validate(step, 
                        val_loader=val_loader, 
                        train_loader=train_loader, 
                        prefix=prefix)
    
    def evaluate(self, step, loader, prefix, time_step_name = "total_epoch", 
                 extra_log = None):
        return super().evaluate(step, 
                                loader, 
                                prefix, 
                                time_step_name=time_step_name, 
                                extra_log=extra_log)

    def visualize(self, total_epoch):
        return super().visualize(total_epoch)

    def gaussian_nll(self, mean_traj: Tensor, 
                     yy: Tensor, 
                     log_sigma: Tensor, 
                     initial_step: Tensor, 
                     reduction: str = 'mean') -> Tensor:
        """mean_traj: shape [B, X, T, C]
        yy: shape [B, X, T, C]
        log_sigma: shape [B, X, T, C]
        initial_step: int
        reduction: str

        return: shape [1]

        """
        if mean_traj.shape != yy.shape:
            raise ValueError(f"""mean_traj shape does not match yy: 
                             mean_traj.shape = {mean_traj.shape},
                             yy.shape = {yy.shape}""")
        if log_sigma.shape != yy.shape:
            raise ValueError(f"""log_sigma shape does not match yy:
                             log_sigma.shape = {log_sigma.shape} 
                             yy.shape = {yy.shape}""")
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi,
                                         device=mean_traj.device, 
                                         dtype=mean_traj.dtype))
        loss = 0.5 * (mean_traj - yy)**2 \
            * torch.exp(-2.0 * log_sigma) + \
                0.5 * log_2pi + log_sigma
        loss = loss[..., initial_step:, :]
        if reduction == 'mean':
            loss = loss.mean()
        else:
            raise ValueError("Only mean reduction is currently supported for gaussian_nll")
        return loss

    def init_training(self, al_iter, load_train_data=True, 
                      warm_start_model_state: OrderedDict | None = None,
                      ):
        self.model_wrapper.init_training(al_iter, 
                                      load_train_data=load_train_data, 
                                      warm_start_model_state=warm_start_model_state)

        self.task_norm = self.model_wrapper.task_norm
        self.loss.init_training(self.task_norm, self.initial_step)
        self.optimizer = self.optimizer_factory(list(self.model_wrapper.model.parameters()))
        self.scheduler = self.scheduler_factory(self.optimizer)
        self.first_iter_nb = self.model_wrapper.first_iter_nb


    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        # print("len train dataset:", len(self.train_loader.dataset))
        # print("len train loader:", len(self.train_loader))
        # print("batch size:", self.batch_size)
        # print("train folders:", self.task.train_data_folders)
        t1_epoch = default_timer()

        self.model_wrapper.train()
        self.task_norm = getattr(self.model_wrapper, "task_norm", None)
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
        profile_batches = 0 # adjust for profiling

        if self.noise_mode == "channel_std_rel" and self.task_norm is not None:
            std = self.noise * self.task_norm.std_channels
        elif self.noise_mode == "constant" or self.task_norm is None:
            std = self.noise
        else:
            raise ValueError(f"Invalid noise mode {self.noise_mode}")

        #loader_iter = iter(self.train_loader)
        #batch_idx = 0

        for batch_idx, (xx, yy, grid, pde_param, t_idx) in enumerate(self.train_loader):
        #while True:
            do_profile = torch.cuda.is_available() and batch_idx < profile_batches
            # t0 = sync_time(do_profile)
            # try: 
            #     xx, yy, grid, pde_param, t_idx = next(loader_iter)
            # except StopIteration:
            #     break
            # t1 = sync_time(do_profile)
            # if do_profile:
            #     print(f"profile batch {batch_idx}: dataloader = {t1 - t0:.4f} s")

            t0 = sync_time(do_profile)
            xx = xx.to(device)
            yy = yy.to(device)
            grid = grid.to(device)
            pde_param = pde_param.to(device)
            t_idx = t_idx.to(device)
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"profile batch {batch_idx}: move to device = {t1 - t0:.4f} s")

            _batch = yy.size(0)
                 
            if self.block_grad:
                xx = xx.detach()
 
            unc_cell = []
            pred = yy[..., :self.initial_step, :]
            # pdebench_loss style behavior to incorporate training schedule
            if self.training_type in ['autoregressive', 
                                        'teacher_forcing',
                                        'teacher_forcing_schedule',
                                        'full_teacher_forcing']:
                
                unc_cell = [torch.zeros_like(xx)]

                t_train = yy.shape[-2]

                for t in range(self.initial_step, yy.shape[-2]):
                    y = yy[..., t:t + 1, :]
                    xx = xx + torch.normal(torch.zeros_like(xx), std)  # add random noise
                    abs_t_idx = t_idx + t - self.initial_step
                    model_pred, unc = self.uncertainty(xx, 
                                                grid, 
                                                pde_param=pde_param, 
                                                t_idx=abs_t_idx,
                                                )
                    pred = torch.cat((pred, model_pred), -2)
                    if self.training_type == "autoregressive":
                        xx = torch.cat((xx[..., 1:, :], model_pred), dim=-2)
                    if self.training_type == "teacher_forcing_schedule":
                        ep_norm = current_epoch / num_epoch
                        t_norm = 0.5 * (1. + math.tanh((ep_norm - 0.5) / 0.2))  # tanh
                        t_ar = int(t_train * t_norm)
                        t_ar = min(t_train, max(1, t_ar))
                        if t < t_ar:
                            xx = torch.cat((xx[..., 1:, :], model_pred), dim=-2)  # auto-regressive
                        else:
                            xx = torch.cat((xx[..., 1:, :], y), dim=-2)  # teacher forcing
                    if self.training_type in {"teacher_forcing", "full_teacher_forcing"}:
                        xx = torch.cat((xx[..., 1:, :], y), dim=-2)
                    unc_cell.append(unc)
                log_sigma = torch.concat(unc_cell, dim=-2)
            elif self.training_type == 'full_trajectory':
                xx = xx + torch.normal(torch.zeros_like(xx), std)
                t0 = sync_time(do_profile)
                pred, log_sigma = self.unc_roll_out_log_sigma(xx, 
                                                        grid, 
                                                        yy.shape[-2],
                                                        pde_param=pde_param,
                                                        t_idx=t_idx,
                                                        return_features=False)
                t1 = sync_time(do_profile)
                if do_profile:
                    print(f"profile batch {batch_idx}: forward_unc = {t1 - t0:.4f} s")
    
            else:
                raise ValueError(self.training_type)
            

            # currently, prediction is denormalized, log_sigma is normalized.
            # So, renormalize to put them on the same scale
            if self.task_norm is not None:
                pred_norm = self.task_norm.norm_traj(pred)
                yy_norm = self.task_norm.norm_traj(yy)
            else:
                pred_norm = pred
                yy_norm = yy

            t0 = sync_time(do_profile)
            loss = self.gaussian_nll(pred_norm, yy_norm, log_sigma, self.initial_step, reduction='mean')
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"profile batch {batch_idx}: loss = {t1 - t0:.4f} s")

            self.optimizer.zero_grad()

            t0 = sync_time(do_profile)
            loss.backward()
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"profile batch {batch_idx}: backward = {t1 - t0:.4f} s")

            t0 = sync_time(do_profile)
            total_norm = 0
            model_parameters = list(self.model_wrapper.model.parameters())

            for param in model_parameters:
                if param.requires_grad and param.grad is not None:
                    grad = param.grad
                    grad_sum += grad.abs().mean().detach().cpu()
                    grad_sum_n += 1
                    max_grad = max(max_grad, grad.abs().max().detach().cpu().item())
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"profile batch {batch_idx}: grad stats = {t1 - t0:.4f} s")
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

            t0 = sync_time(do_profile)
            self.optimizer.step()
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"batch profile {batch_idx}: optimizer step = {t1 - t0:.4f} s")

            t0 = sync_time(do_profile)
            all_se = batch_errors(pred, yy, self.initial_step)
            train_mse_sum += all_se.mean().detach() * _batch
            train_max_l2 = max(torch.sqrt(all_se).max().detach().item(), train_max_l2)
            train_loss_sum += loss.detach().item() * _batch
            n += _batch
            t1 = sync_time(do_profile)
            if do_profile:
                print(f"batch profile {batch_idx}: train metrics = {t1 - t0:.4f} s")

            if self.step_batch != -1 and tb_idx % self.step_batch == 0 and tb_idx < self.first_iter_nb * num_epoch:
                self.scheduler.step()
            #batch_idx += 1

        train_rmse = torch.sqrt(train_mse_sum / n).item()
        train_avg_loss = train_loss_sum / n
        t2_epoch = default_timer()
        avg_grad = grad_sum / grad_sum_n
        if self.step_batch == -1:
            self.scheduler.step()

        print("epoch:", str(current_epoch), "time:", np.round(t2_epoch - t1_epoch, 2),
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
                      warm_start_model_states: OrderedDict | None = None,
                      #warm_start_unc_model_state: OrderedDict | None = None
                      ):
        total_time = 0
        self.init_training(al_iter, 
                           load_train_data=True,
                           warm_start_model_state=warm_start_model_states,
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
        return {"optimizer": self.optimizer.state_dict(), 
                "scheduler": self.scheduler.state_dict()}

    def set_extra_state(self, state):
        self.optimizer.load_state_dict(state["optimizer"])
        self.scheduler.load_state_dict(state["scheduler"])

def build_mean_unc(task, cfg):
    if task.spatial_dim != 1:
        raise NotImplementedError("Only 1D spatial tasks supported currently.")
    name = cfg.get("name", "mean_unc_model")
    model_wrapper = build_wrapper(task, cfg.model_wrapper, name=name + "/fno")

    return MeanUnc(model_wrapper,
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
                        cfg.model_wrapper.model.if_conditional,
                        task,
                        cfg.model_wrapper.training_type,
                        cfg.model_wrapper.t_train,
                        batch_size = cfg.model_wrapper.batch_size,
                        val_period = cfg.model_wrapper.val_period,
                        vis_period = cfg.model_wrapper.vis_period,
                        loss = model_wrapper.loss,
                    )





    
    


        
                 
