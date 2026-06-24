import unittest

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from unittest.mock import patch

from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.prob_models.mean_unc_model import MeanUncModel
from al4pde.prob_models.unc_model import UncertaintyModelFullTrajectory1d


class DummySim:
    spatial_dim = 1
    num_channels = 1
    channel_names = ["u"]
    pde_name = "dummy"
    pde_param_names = ["nu"]


class DummyTask:
    initial_step = 2
    reduced_resolution = 1
    reduced_resolution_t = 1
    skip_initial_steps = 0
    reduced_batch = 1
    img_save_path = "/tmp"
    num_pde_param = 1
    num_channels = 1
    spatial_dim = 1
    sim = DummySim()

    def get_grid(self, n=1):
        return torch.linspace(0, 1, 8).reshape(1, 8, 1)


class DummyLoss(nn.Module):
    def init_training(self, task_norm, initial_step):
        self.task_norm = task_norm
        self.initial_step = initial_step

    def forward(self, pred, target, reduction="mean", initial_step=0):
        loss = (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2
        if reduction == "mean":
            return loss.mean()
        if reduction is None:
            return loss
        raise ValueError(reduction)


class DummyMeanModel(nn.Module):
    def __init__(self, pred):
        super().__init__()
        self.pred = pred
        self.val_loader = object()
        self.train_loader = object()
        self.train_loader_full_traj = object()
        self.reduced_inputs = None

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        if return_features:
            return self.pred, self.pred
        return self.pred

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        out = self.pred[..., :final_step, :]
        if return_features:
            return out, out
        return out

    def reduce_input_res(self, xx, grid):
        self.reduced_inputs = (xx, grid)
        return xx[..., ::2, :] if False else (xx, grid)

    def visualize(self, total_epoch):
        return total_epoch


class FixedLogSigmaModel(nn.Module):
    def __init__(self, log_sigma):
        super().__init__()
        self.log_sigma = log_sigma

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        return self.log_sigma


class RecordingLogSigmaModel(nn.Module):
    def __init__(self, log_sigma):
        super().__init__()
        self.log_sigma = log_sigma
        self.inputs = None

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        self.inputs = (xx, grid, pde_param, t_idx)
        return self.log_sigma


class OffsetNormalizer:
    def norm_traj(self, xx):
        return xx + 1.0

    def norm_grid(self, grid):
        return grid + 2.0

    def norm_param(self, pde_param):
        return pde_param + 3.0


class DummyScheduler:
    def __init__(self):
        self.num_steps = 0

    def step(self):
        self.num_steps += 1

    def get_lr(self):
        return [0.1]


class TrainableMeanModel(nn.Module):
    def __init__(self, train_loader):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.model.weight, 0.25)
        self.train_loader = train_loader
        self.train_loader_full_traj = train_loader
        self.val_loader = []
        self.task_norm = None
        self.first_iter_nb = len(train_loader)

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        one_step = self.model(xx[..., :1, :])
        return one_step.repeat(*([1] * (one_step.dim() - 2)), final_step, 1)


class BuildableTrainableMeanModel(TrainableMeanModel):
    def __init__(self, train_loader):
        super().__init__(train_loader)
        self.loss = DummyLoss()
        self.init_training_calls = []

    def init_training(self, al_iter, load_train_data=True, warm_start_model_state=None):
        self.init_training_calls.append((al_iter, load_train_data, warm_start_model_state))
        self.task_norm = None
        self.first_iter_nb = len(self.train_loader)


class TrainableLogSigmaModel(nn.Module):
    def __init__(self, traj_len):
        super().__init__()
        self.log_sigma = nn.Parameter(torch.zeros(1, traj_len))

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        return self.log_sigma.expand(xx.shape[0], -1)

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None):
        return self.forward(xx, grid, pde_param, t_idx)[..., :final_step]


def make_mean_unc_model(mean_model, unc_model):
    model = MeanUncModel.__new__(MeanUncModel)
    nn.Module.__init__(model)
    model.mean_model = mean_model
    model.unc_model = unc_model
    model.if_conditional = True
    return model


class MeanUncertaintyModelTest(unittest.TestCase):
    def test_uncertainty_model_builds_conv1d_input_channels(self):
        model = UncertaintyModelFullTrajectory1d(
            input_channels=5,
            hidden_units=4,
            output_channels=3,
            input_grid_dim=4,
            task=DummyTask(),
            training_type="full_trajectory",
            t_train=5,
            batch_size=2,
            val_period=1,
            vis_period=1,
            loss=DummyLoss(),
        )
        xx = torch.arange(16, dtype=torch.float32).reshape(2, 4, 2, 1)
        grid = torch.tensor([[0.0], [0.25], [0.5], [0.75]])
        pde_param = torch.tensor([[0.1], [0.2]])

        conv_input = model.build_conv_input(xx, grid, pde_param=pde_param)

        self.assertEqual(conv_input.shape, (2, 4, 4))
        self.assertTrue(torch.equal(conv_input[:, :2], torch.flatten(xx, start_dim=-2).permute(0, 2, 1)))
        self.assertTrue(torch.equal(conv_input[:, 2], grid[:, 0].expand(2, -1)))
        self.assertTrue(torch.equal(conv_input[0, 3], torch.full((4,), 0.1)))
        self.assertTrue(torch.equal(conv_input[1, 3], torch.full((4,), 0.2)))

    def test_uncertainty_model_roll_out_returns_log_sigma_horizon(self):
        model = UncertaintyModelFullTrajectory1d(
            input_channels=3,
            hidden_units=4,
            output_channels=5,
            input_grid_dim=8,
            task=DummyTask(),
            training_type="full_trajectory",
            t_train=5,
            batch_size=2,
            val_period=1,
            vis_period=1,
            loss=DummyLoss(),
        )

        log_sigma = model.roll_out(
            torch.randn(2, 8, 1, 1),
            torch.randn(2, 8, 1),
            final_step=3,
            pde_param=torch.randn(2, 1),
        )

        self.assertEqual(log_sigma.shape, (2, 3))

    def test_uncertainty_model_rejects_wrong_grid_resolution(self):
        model = UncertaintyModelFullTrajectory1d(
            input_channels=2,
            hidden_units=4,
            output_channels=3,
            input_grid_dim=8,
            task=DummyTask(),
            training_type="full_trajectory",
            t_train=5,
            batch_size=2,
            val_period=1,
            vis_period=1,
            loss=DummyLoss(),
        )

        with self.assertRaisesRegex(ValueError, "expected X=8"):
            model(
                torch.randn(2, 6, 1, 1),
                torch.randn(2, 6, 1),
            )

    def test_uncertainty_only_unc_roll_out_is_not_supported(self):
        model = UncertaintyModelFullTrajectory1d(
            input_channels=2,
            hidden_units=4,
            output_channels=3,
            input_grid_dim=8,
            task=DummyTask(),
            training_type="full_trajectory",
            t_train=5,
            batch_size=2,
            val_period=1,
            vis_period=1,
            loss=DummyLoss(),
        )

        with self.assertRaises(NotImplementedError):
            model.unc_roll_out(torch.randn(2, 8, 1, 1), torch.randn(2, 8, 1), 3)

    def test_mean_unc_roll_out_expands_sigma_to_trajectory_shape(self):
        mean = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5, 1)
        log_sigma = torch.log(torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]]))
        model = make_mean_unc_model(DummyMeanModel(mean), FixedLogSigmaModel(log_sigma))

        mean_traj, unc_traj = model.unc_roll_out(
            torch.zeros(1, 2, 1, 1),
            torch.zeros(1, 2, 1),
            final_step=3,
        )

        self.assertTrue(torch.equal(mean_traj, mean[..., :3, :]))
        self.assertEqual(unc_traj.shape, (1, 2, 3, 1))
        self.assertTrue(torch.equal(unc_traj[0, 0, :, 0], torch.tensor([1.0, 2.0, 3.0])))
        self.assertTrue(torch.equal(unc_traj[0, 1, :, 0], torch.tensor([1.0, 2.0, 3.0])))

    def test_mean_unc_roll_out_rejects_short_uncertainty_horizon(self):
        mean = torch.zeros(1, 2, 5, 1)
        log_sigma = torch.zeros(1, 2)
        model = make_mean_unc_model(DummyMeanModel(mean), FixedLogSigmaModel(log_sigma))

        with self.assertRaisesRegex(ValueError, "Uncertainty trajectory is shorter"):
            model.unc_roll_out(
                torch.zeros(1, 2, 1, 1),
                torch.zeros(1, 2, 1),
                final_step=3,
            )

    def test_mean_unc_delegates_loaders_and_resolution_to_mean_model(self):
        mean_model = DummyMeanModel(torch.zeros(1, 2, 3, 1))
        model = make_mean_unc_model(mean_model, FixedLogSigmaModel(torch.zeros(1, 3)))
        xx = torch.zeros(1, 2, 1, 1)
        grid = torch.zeros(1, 2, 1)

        self.assertIs(model.val_loader, mean_model.val_loader)
        self.assertIs(model.train_loader, mean_model.train_loader)
        self.assertIs(model.train_loader_full_traj, mean_model.train_loader_full_traj)
        self.assertEqual(model.reduce_input_res(xx, grid), (xx, grid))
        self.assertEqual(mean_model.reduced_inputs, (xx, grid))

    def test_mean_unc_roll_out_normalizes_uncertainty_inputs_when_available(self):
        mean_model = DummyMeanModel(torch.zeros(1, 2, 3, 1))
        mean_model.task_norm = OffsetNormalizer()
        unc_model = RecordingLogSigmaModel(torch.zeros(1, 3))
        model = make_mean_unc_model(mean_model, unc_model)
        xx = torch.zeros(1, 2, 1, 1)
        grid = torch.zeros(1, 2, 1)
        pde_param = torch.zeros(1, 1)
        t_idx = torch.zeros(1)

        model.unc_roll_out(xx, grid, final_step=3, pde_param=pde_param, t_idx=t_idx)

        xx_seen, grid_seen, pde_param_seen, t_idx_seen = unc_model.inputs
        self.assertTrue(torch.equal(xx_seen, xx + 1.0))
        self.assertTrue(torch.equal(grid_seen, grid + 2.0))
        self.assertTrue(torch.equal(pde_param_seen, pde_param + 3.0))
        self.assertIs(t_idx_seen, t_idx)

    def test_gaussian_nll_matches_log_sigma_formula(self):
        model = make_mean_unc_model(
            DummyMeanModel(torch.zeros(1, 2, 3, 1)),
            FixedLogSigmaModel(torch.zeros(1, 3)),
        )
        mean = torch.zeros(1, 2, 3, 1)
        yy = torch.ones(1, 2, 3, 1)
        log_sigma = torch.zeros(1, 3)

        loss = model.gaussian_nll(mean, yy, log_sigma, initial_step=1)

        expected = 0.5 + 0.5 * torch.log(torch.tensor(2.0 * torch.pi))
        self.assertTrue(torch.allclose(loss, expected))

    def test_train_single_epoch_updates_mean_and_uncertainty_parameters(self):
        xx = torch.ones(2, 4, 1, 1)
        yy = torch.ones(2, 4, 3, 1) * 2.0
        grid = torch.linspace(0, 1, 4).reshape(1, 4, 1).repeat(2, 1, 1)
        pde_param = torch.zeros(2, 1)
        t_idx = torch.zeros(2)
        train_loader = [(xx, yy, grid, pde_param, t_idx)]
        mean_model = TrainableMeanModel(train_loader)
        unc_model = TrainableLogSigmaModel(traj_len=3)
        model = make_mean_unc_model(mean_model, unc_model)
        model.noise = 0.0
        model.noise_mode = "constant"
        model.block_grad = False
        model.max_grad_norm = None
        model.step_batch = -1
        model.norm_max_counter = 0
        model.max_norm_stat = 0
        model.first_iter_nb = len(train_loader)
        model.name = "mean_unc_test"
        model.val_period = 999
        model.initial_step = 1
        model.task_norm = None
        params = list(mean_model.model.parameters()) + list(unc_model.parameters())
        model.optimizer = torch.optim.SGD(params, lr=0.1)
        model.scheduler = DummyScheduler()

        mean_weight_before = mean_model.model.weight.detach().clone()
        log_sigma_before = unc_model.log_sigma.detach().clone()

        model.train_single_epoch(current_epoch=1, total_epoch=1, num_epoch=2)

        self.assertIsNotNone(mean_model.model.weight.grad)
        self.assertIsNotNone(unc_model.log_sigma.grad)
        self.assertFalse(torch.equal(mean_model.model.weight.detach(), mean_weight_before))
        self.assertFalse(torch.equal(unc_model.log_sigma.detach(), log_sigma_before))
        self.assertEqual(model.scheduler.num_steps, 1)

    def test_build_prob_model_mean_unc_initializes_and_trains_one_epoch(self):
        xx = torch.ones(2, 8, 2, 1)
        yy = torch.ones(2, 8, 5, 1) * 2.0
        grid = torch.linspace(0, 1, 8).reshape(1, 8, 1).repeat(2, 1, 1)
        pde_param = torch.zeros(2, 1)
        t_idx = torch.zeros(2)
        train_loader = [(xx, yy, grid, pde_param, t_idx)]
        mean_model = BuildableTrainableMeanModel(train_loader)
        cfg = OmegaConf.create({
            "_target_": "al4pde.prob_models.mean_unc_model.MeanUncModel",
            "model_wrapper": {
                "_target_": "unused.because.build_wrapper.is.mocked",
                "training_type": "full_trajectory",
                "t_train": 5,
                "batch_size": 2,
                "val_period": 999,
                "vis_period": 999,
                "noise": 0.0,
                "noise_mode": "constant",
                "block_grad": False,
                "max_grad_norm": None,
                "step_batch": -1,
                "optimizer": {
                    "_target_": "torch.optim.SGD",
                    "lr": 0.1,
                },
                "scheduler": {
                    "_target_": "torch.optim.lr_scheduler.StepLR",
                    "step_size": 100,
                    "gamma": 1.0,
                },
                "model": {
                    "traj_len": 5,
                },
            },
            "unc_model": {
                "hidden_units": 4,
                "distribution": "normal",
                "if_conditional": True,
            },
        })

        with patch("al4pde.prob_models.mean_unc_model.build_wrapper", return_value=mean_model):
            model = build_prob_model(DummyTask(), cfg)

        self.assertIs(model.mean_model, mean_model)
        self.assertIsNone(model.unc_model)

        model.init_training(al_iter=0)

        self.assertIsNotNone(model.unc_model)
        self.assertEqual(model.unc_model.input_grid_dim, 8)
        self.assertEqual(model.unc_model.block_1[0].in_channels, 4)

        mean_weight_before = mean_model.model.weight.detach().clone()
        unc_param_before = next(model.unc_model.parameters()).detach().clone()

        model.train_single_epoch(current_epoch=1, total_epoch=1, num_epoch=1)

        mean_traj, unc_traj = model.unc_roll_out(xx, grid, final_step=5, pde_param=pde_param, t_idx=t_idx)

        self.assertEqual(mean_traj.shape, (2, 8, 5, 1))
        self.assertEqual(unc_traj.shape, (2, 8, 5, 1))
        self.assertFalse(torch.equal(mean_model.model.weight.detach(), mean_weight_before))
        self.assertFalse(torch.equal(next(model.unc_model.parameters()).detach(), unc_param_before))


if __name__ == "__main__":
    unittest.main()
