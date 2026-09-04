import unittest
import sys
import types
from unittest.mock import patch

import torch
import torch.nn as nn

if "tensordict" not in sys.modules:
    tensordict_stub = types.ModuleType("tensordict")
    tensordict_stub.TensorDict = object
    sys.modules["tensordict"] = tensordict_stub

from al4pde.acquisition.model_change_based import ModelChangeBased
from al4pde.prob_models.mean_unc_model import MeanUncModel


class DummyLoss(nn.Module):
    def forward(self, pred, target, reduction="mean", initial_step=0):
        loss = (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2
        if reduction == "mean":
            return loss.mean()
        if reduction is None:
            return loss
        raise ValueError(reduction)


class DummyMeanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.model.weight, 0.25)
        self.task_norm = None

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        one_step = self.model(xx[..., :1, :])
        return one_step.repeat(*([1] * (one_step.dim() - 2)), final_step, 1)


class DummyUncModel(nn.Module):
    def __init__(self, traj_len):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1, traj_len))

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, time_coord=None):
        return self.log_sigma.expand(xx.shape[0], -1)[..., :final_step]


class DummyTask:
    spatial_dim = 1


class RecordingMeanUncModel(MeanUncModel):
    def __init__(self, scores):
        nn.Module.__init__(self)
        self.scores = list(scores)
        self.calls = []

    def reduce_input_res(self, xx, grid):
        return xx, grid

    def expected_model_change(self, xx, grid, final_step, pde_param, t_idx, num_mc_samples, use_nll_loss):
        self.calls.append((xx.clone(), grid.clone(), pde_param.clone(), t_idx.clone(), final_step))
        return torch.tensor(self.scores.pop(0), device=xx.device)


def make_batch_level_acquisition(selection_mode="top_k"):
    acq = ModelChangeBased.__new__(ModelChangeBased)
    acq.batch_size = 2
    acq.pred_batch_size = 2
    acq.selection_mode = selection_mode
    acq.num_mc_samples = 3
    acq.use_nll_loss = True
    acq.power_beta = 1
    acq.unc_num_rollout_steps = 4
    acq.data_schedule = lambda al_iter: 1
    return acq


class ModelChangeBasedTest(unittest.TestCase):
    def test_top_k_selects_all_samples_from_highest_emc_batch(self):
        acq = make_batch_level_acquisition("top_k")
        prob_model = RecordingMeanUncModel(scores=[0.1, 0.9, 0.2])
        ic_pool = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
        pde_param_pool = torch.arange(6, dtype=torch.float32).reshape(6, 1)
        grid = torch.zeros(1, 1)

        with patch("al4pde.acquisition.model_change_based.wandb.log") as wandb_log:
            sel_idx = acq.select_next(
                prob_model,
                ic_pool,
                pde_param_pool,
                ic_train=torch.empty(0),
                pde_param_train=torch.empty(0),
                grid=grid,
                al_iter=0,
            )

        self.assertEqual(sel_idx, [2, 3])
        self.assertEqual(len(prob_model.calls), 3)
        self.assertEqual([call[0].shape[0] for call in prob_model.calls], [2, 2, 2])
        self.assertTrue(torch.equal(prob_model.calls[1][0], ic_pool[2:4]))
        logged = wandb_log.call_args.args[0]
        self.assertTrue(torch.equal(logged["al/acq_avg_emc"], torch.tensor(0.9)))

    def test_top_k_selects_multiple_batches_when_schedule_requests_more_than_one(self):
        acq = make_batch_level_acquisition("top_k")
        acq.data_schedule = lambda al_iter: 2
        prob_model = RecordingMeanUncModel(scores=[0.1, 0.9, 0.2])
        ic_pool = torch.arange(6, dtype=torch.float32).reshape(6, 1, 1, 1)
        pde_param_pool = torch.arange(6, dtype=torch.float32).reshape(6, 1)
        grid = torch.zeros(1, 1)

        with patch("al4pde.acquisition.model_change_based.wandb.log"):
            sel_idx = acq.select_next(
                prob_model,
                ic_pool,
                pde_param_pool,
                ic_train=torch.empty(0),
                pde_param_train=torch.empty(0),
                grid=grid,
                al_iter=0,
            )

        self.assertEqual(sel_idx, [2, 3, 4, 5])

    def test_expected_model_change_runs_multiple_mc_samples_and_returns_scalar(self):
        model = MeanUncModel.__new__(MeanUncModel)
        nn.Module.__init__(model)
        model.mean_model = DummyMeanModel()
        model.unc_model = DummyUncModel(traj_len=4)
        model.task = DummyTask()
        model.initial_step = 1
        model.if_conditional = True
        model.time_coord_mode = "abs_time"
        model.default_unc_traj_len = None
        model.loss = DummyLoss()

        xx = torch.ones(2, 3, 1, 1)
        grid = torch.linspace(0, 1, 3).reshape(1, 3, 1).repeat(2, 1, 1)
        pde_param = torch.zeros(2, 1)
        t_idx = torch.zeros(2)

        score = model.expected_model_change(
            xx,
            grid,
            final_step=4,
            pde_param=pde_param,
            t_idx=t_idx,
            num_mc_samples=3,
            use_nll_loss=True,
        )

        self.assertEqual(score.shape, torch.Size([]))
        self.assertTrue(torch.isfinite(score))
        self.assertGreater(score.item(), 0.0)


if __name__ == "__main__":
    unittest.main()
