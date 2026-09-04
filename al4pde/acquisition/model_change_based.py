import torch
import wandb
import math
from torch.utils.data import TensorDataset, DataLoader
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from al4pde.acquisition.pool_based import PoolBased
from al4pde.prob_models.prob_model import ProbModel
from al4pde.acquisition.uncertainty_based import top_k, power_sampling, random
from al4pde.prob_models.mean_unc_model import MeanUncModel
from al4pde.prob_models.mean_unc_internal import MeanUnc

class ModelChangeBased(PoolBased):
    def __init__(self, task, data_schedule, batch_size, pool_size, unc_eval_mode,
                     unc_num_rollout_steps_rel, selection_mode, num_mc_samples, use_nll_loss, power_beta=1, pred_batch_size=128):
            super().__init__(task, data_schedule, batch_size, pool_size, unc_eval_mode, unc_num_rollout_steps_rel,
                             pred_batch_size=pred_batch_size)
            self.selection_mode = selection_mode
            assert selection_mode in ["random", "top_k", "power"]
            self.power_beta = power_beta
            self.num_mc_samples = num_mc_samples
            self.use_nll_loss = use_nll_loss
            if not self.batch_size == self.pred_batch_size:
                raise ValueError("Currently, acquisition batch size must match"\
                "pred_batch_size due to the expected model change method" \
                "ranking and acquiring entire batches rather than individual" \
                "candidates.")

    def generate(self, prob_model: ProbModel, al_iter: int, 
                 train_loader: DataLoader, 
                 enable_grads: bool = False) -> None:
        super().generate(prob_model, 
                         al_iter, 
                         train_loader=train_loader,
                         enable_grads=True)
    
    def select_next(self, prob_model: ProbModel, ic_pool: torch.Tensor, pde_param_pool: torch.Tensor,
                    ic_train: torch.Tensor, pde_param_train: torch.Tensor, grid: torch.Tensor, al_iter: int,
                    train_loader=None) -> torch.Tensor:
        if not isinstance(prob_model, (MeanUnc, MeanUncModel)):
            raise TypeError("Acquisition with ModelChangeBased methods" \
            "currently require MeanUnc or MeanUncModel as the ProbModel.")
        dataset = TensorDataset(ic_pool, pde_param_pool)
        emc_list = []
        if self.selection_mode != "random":
            loader = DataLoader(dataset, batch_size=self.pred_batch_size)
            grid = grid.to(device)
            for i, batch in enumerate(loader):
                ic = batch[0].to(device)
                pde_param = batch[1].to(device)
                grid_batch = grid.expand([len(ic), ] + list(grid.shape))
                emc_batch = self.expected_model_change(prob_model, 
                                                       ic, 
                                                       grid_batch, 
                                                       pde_param, 
                                                       self.num_mc_samples,
                                                       self.use_nll_loss) # shape [1], one number per batch
                emc_list.append(emc_batch)
            emc = torch.stack(emc_list, dim=0) # shape [num batches in dataset]
        n_samples = self.num_batches(al_iter) * self.batch_size
        # top_k and power_sampling here use expected model change instead of unc;
        # they are general functions that only need positive scores
        if self.selection_mode == "top_k":
            # need to select all indices from the corresponding batch selected
            batch_size = loader.batch_size
            sel_batch_idx = top_k(emc, math.ceil(n_samples / batch_size)) # gives loader indices, i.e. indices of the batches in the loader

            sel_idx = []
            for b in sel_batch_idx:
                b = int(b.item())
                start = b * batch_size
                end = min((b + 1) * batch_size, len(loader.dataset))
                sel_idx.extend(range(start, end))
            sel_idx = sel_idx[:n_samples]
        elif self.selection_mode == "power":
            batch_size = loader.batch_size
            sel_batch_idx = power_sampling(emc, math.ceil(n_samples / batch_size), self.power_beta)
            
            sel_idx = []
            for b in sel_batch_idx:
                b = int(b.item())
                start = b * batch_size
                end = min((b + 1) * batch_size, len(loader.dataset))
                sel_idx.extend(range(start, end))
            sel_idx = sel_idx[:n_samples]
        elif self.selection_mode == "random":
            sel_idx = random(len(ic_pool), n_samples)
        else:
            raise ValueError(self.selection_mode)
        if self.selection_mode != "random":
            wandb.log({"al/acq_avg_emc": emc[sel_batch_idx].mean(), "al/al_iter": al_iter})
        return sel_idx

    @property
    def name(self):
        return "emc_" + self.selection_mode
    
