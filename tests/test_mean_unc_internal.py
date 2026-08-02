import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from al4pde.models.fno_wrapper import build_uncertainty_spec
from al4pde.modules.fno import FNO1d
from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.prob_models.mean_unc_internal import MeanUnc


class DummySim:
    autonomous = True


class DummyParamGen:
    log_scale = [False]


class DummyTask:
    initial_step = 1
    reduced_resolution = 1
    reduced_resolution_t = 1
    skip_initial_steps = 0
    reduced_batch = 1
    img_save_path = "/tmp"
    num_pde_param = 1
    num_channels = 1
    spatial_dim = 1
    sim = DummySim()
    param_gen = DummyParamGen()


class DummyLoss(nn.Module):
    name = "dummy_loss"

    def init_training(self, task_norm, initial_step):
        self.task_norm = task_norm
        self.initial_step = initial_step

    def forward(self, pred, target, reduction="mean", initial_step=0):
        loss = (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2
        if reduction == "mean":
            return loss.mean()
        if reduction == "mean_traj":
            return loss.flatten(1, -1).mean(1)
        if reduction is None:
            return loss
        raise ValueError(reduction)


class DummyScheduler:
    def __init__(self, optimizer=None):
        self.optimizer = optimizer
        self.num_steps = 0

    def step(self):
        self.num_steps += 1

    def get_lr(self):
        return [0.1]

    def state_dict(self):
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state):
        self.num_steps = state["num_steps"]


class TinyMeanUncModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.mean_weight = nn.Parameter(torch.tensor(0.25))
        self.log_sigma = nn.Parameter(torch.tensor([[0.0]]))

    def forward(self, xx):
        return self.mean_weight * xx[..., -1:, :]


class DummyInternalWrapper(nn.Module):
    def __init__(self, train_loader=None):
        super().__init__()
        self.model = TinyMeanUncModule()
        self.loss = DummyLoss()
        self.task_norm = None
        self.first_iter_nb = len(train_loader) if train_loader is not None else 1
        self._train_loader = train_loader
        self._val_loader = []
        self.train_loader_full_traj = train_loader
        self.init_calls = []
        self.t_idx_seen = []

    @property
    def train_loader(self):
        return self._train_loader

    @property
    def val_loader(self):
        return self._val_loader

    def init_training(self, al_iter, load_train_data=True, warm_start_model_state=None):
        self.init_calls.append((al_iter, load_train_data, warm_start_model_state))
        if warm_start_model_state is not None:
            self.model.load_state_dict(warm_start_model_state)

    def forward_mean_unc(self, xx, grid, pde_param=None, t_idx=None):
        self.t_idx_seen.append(t_idx.detach().clone() if t_idx is not None else None)
        mean = self.model(xx)
        log_sigma = self.model.log_sigma.expand(xx.shape[0], -1)
        return mean, log_sigma

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        return self.model(xx)

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        pred = xx
        for _ in range(xx.shape[-2], final_step):
            step = self.model(pred)
            pred = torch.cat((pred, step), dim=-2)
        return pred

    def one_step_pred(self, yy, grid, param, t_idx=None):
        return yy

    def reduce_input_res(self, xx, grid):
        return xx, grid


def make_mean_unc(wrapper, training_type="autoregressive"):
    return MeanUnc(
        wrapper,
        noise=0.0,
        noise_mode="constant",
        block_grad=False,
        max_grad_norm=None,
        step_batch=-1,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        scheduler_factory=lambda optimizer: DummyScheduler(optimizer),
        name="mean_unc_internal_test",
        if_conditional=True,
        task=DummyTask(),
        training_type=training_type,
        t_train=3,
        batch_size=2,
        val_period=999,
        vis_period=999,
        loss=DummyLoss(),
    )


class MeanUncInternalTest(unittest.TestCase):
    def test_uncertainty_spec_builds_enabled_taps_only(self):
        cfg = OmegaConf.create({
            "enabled": True,
            "detach_features": False,
            "clamp_log_sigma": True,
            "min_log_sigma": -3.0,
            "max_log_sigma": 2.0,
            "taps": [
                {
                    "name": "skip_me",
                    "enabled": False,
                    "block_idx": 0,
                    "head": {"in_channels": 4, "out_channels": 8},
                },
                {
                    "name": "block1",
                    "enabled": True,
                    "block_idx": 1,
                    "head": {
                        "in_channels": 4,
                        "hidden_channels": 5,
                        "out_channels": 6,
                        "num_layers": 2,
                        "use_end_activation": True,
                    },
                },
                {
                    "name": "last",
                    "enabled": True,
                    "block_idx": 6,
                    "head": {
                        "hidden_channels": 5,
                        "out_channels": 1,
                        "num_layers": 1,
                        "use_end_activation": False,
                    },
                },
            ],
        })

        spec = build_uncertainty_spec(cfg)

        self.assertTrue(spec.enabled)
        self.assertTrue(spec.clamp_log_sigma)
        self.assertEqual([tap.name for tap in spec.taps], ["block1", "last"])
        self.assertEqual(spec.taps[0].in_channels, 4)
        self.assertFalse(spec.taps[1].use_end_activation)

    def test_fno1d_returns_mean_and_log_sigma_in_one_forward(self):
        spec = build_uncertainty_spec(OmegaConf.create({
            "enabled": True,
            "taps": [
                {
                    "name": "block1",
                    "enabled": True,
                    "block_idx": 1,
                    "head": {
                        "in_channels": 4,
                        "hidden_channels": 5,
                        "out_channels": 3,
                        "num_layers": 1,
                    },
                },
                {
                    "name": "block5",
                    "enabled": True,
                    "block_idx": 5,
                    "head": {
                        "in_channels": 128,
                        "hidden_channels": 5,
                        "out_channels": 3,
                        "num_layers": 1,
                        "make_channel_first_input": True,
                    },
                },
                {
                    "name": "last",
                    "enabled": True,
                    "block_idx": 6,
                    "head": {
                        "hidden_channels": 5,
                        "out_channels": 1,
                        "num_layers": 1,
                        "use_end_activation": False,
                    },
                },
            ],
        }))
        model = FNO1d(num_channels=1, modes=2, width=4, initial_step=1, uncertainty_spec=spec)
        xx = torch.randn(2, 8, 1)
        grid = torch.linspace(0, 1, 8).reshape(1, 8, 1).repeat(2, 1, 1)

        mean, log_sigma = model(xx, grid, return_uncertainty=True)
        loss = mean.mean() + log_sigma.mean()
        loss.backward()

        self.assertEqual(mean.shape, (2, 8, 1, 1))
        self.assertEqual(log_sigma.shape, (2, 1))
        self.assertIsNotNone(model.unc_heads["block1"].fcs.fc_layers[0].weight.grad)
        self.assertIsNotNone(model.unc_heads["last"].fc_layers[0].weight.grad)

    def test_uncertainty_expands_log_sigma_to_one_step_mean_shape(self):
        wrapper = DummyInternalWrapper()
        model = make_mean_unc(wrapper)
        xx = torch.ones(2, 4, 1, 1)
        grid = torch.zeros(2, 4, 1)

        mean, log_sigma = model.uncertainty(xx, grid, pde_param=torch.zeros(2, 1), t_idx=torch.zeros(2))

        self.assertEqual(mean.shape, (2, 4, 1, 1))
        self.assertEqual(log_sigma.shape, mean.shape)
        self.assertTrue(torch.equal(log_sigma, torch.zeros_like(mean)))

    def test_unc_roll_out_returns_sigma_and_zero_initial_uncertainty(self):
        wrapper = DummyInternalWrapper()
        with torch.no_grad():
            wrapper.model.log_sigma.fill_(torch.log(torch.tensor(2.0)))
        model = make_mean_unc(wrapper)
        xx = torch.ones(2, 4, 1, 1)
        grid = torch.zeros(2, 4, 1)

        pred, unc = model.unc_roll_out(xx, grid, final_step=3, pde_param=torch.zeros(2, 1), t_idx=torch.zeros(2))

        self.assertEqual(pred.shape, (2, 4, 3, 1))
        self.assertEqual(unc.shape, pred.shape)
        self.assertTrue(torch.equal(unc[..., :1, :], torch.zeros_like(unc[..., :1, :])))
        self.assertTrue(torch.allclose(unc[..., 1:, :], torch.full_like(unc[..., 1:, :], 2.0)))

    def test_training_step_updates_mean_and_uncertainty_head_parameters(self):
        xx = torch.ones(2, 4, 1, 1)
        yy = torch.ones(2, 4, 3, 1) * 2.0
        grid = torch.zeros(2, 4, 1)
        pde_param = torch.zeros(2, 1)
        t_idx = torch.zeros(2)
        train_loader = [(xx, yy, grid, pde_param, t_idx)]
        wrapper = DummyInternalWrapper(train_loader)
        model = make_mean_unc(wrapper, training_type="teacher_forcing_schedule")
        model.init_training(al_iter=0)

        mean_before = wrapper.model.mean_weight.detach().clone()
        log_sigma_before = wrapper.model.log_sigma.detach().clone()

        model.train_single_epoch(current_epoch=1, total_epoch=1, num_epoch=2)

        self.assertIsNotNone(wrapper.model.mean_weight.grad)
        self.assertIsNotNone(wrapper.model.log_sigma.grad)
        self.assertFalse(torch.equal(wrapper.model.mean_weight.detach(), mean_before))
        self.assertFalse(torch.equal(wrapper.model.log_sigma.detach(), log_sigma_before))
        self.assertEqual(model.scheduler.num_steps, 1)
        self.assertEqual([t.tolist() for t in wrapper.t_idx_seen], [[0.0, 0.0], [1.0, 1.0]])

    def test_build_prob_model_mean_unc_internal_uses_nested_wrapper(self):
        train_loader = [(torch.ones(1, 4, 1, 1), torch.ones(1, 4, 2, 1), torch.zeros(1, 4, 1),
                         torch.zeros(1, 1), torch.zeros(1))]
        wrapper = DummyInternalWrapper(train_loader)
        cfg = OmegaConf.create({
            "_target_": "al4pde.prob_models.mean_unc_internal.MeanUnc",
            "model_wrapper": {
                "_target_": "al4pde.models.fno_wrapper.FNOWrapper",
                "training_type": "autoregressive",
                "t_train": 3,
                "batch_size": 1,
                "val_period": 999,
                "vis_period": 999,
                "noise": 0.0,
                "noise_mode": "constant",
                "block_grad": False,
                "max_grad_norm": None,
                "step_batch": -1,
                "optimizer": {"_target_": "torch.optim.SGD", "lr": 0.1},
                "scheduler": {"_target_": "torch.optim.lr_scheduler.StepLR", "step_size": 100},
            },
            "unc_model": {"if_conditional": True},
        })

        with patch("al4pde.prob_models.mean_unc_internal.build_wrapper", return_value=wrapper):
            model = build_prob_model(DummyTask(), cfg)

        self.assertIsInstance(model, MeanUnc)
        self.assertIs(model.model_wrapper, wrapper)
        model.init_training(al_iter=0)
        self.assertEqual(wrapper.init_calls, [(0, True, None)])


if __name__ == "__main__":
    unittest.main()
