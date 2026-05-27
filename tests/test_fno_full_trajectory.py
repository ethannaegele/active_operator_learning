import unittest

import torch
import torch.nn as nn

from al4pde.models.fno_wrapper import FNOFullTrajectoryWrapper
from al4pde.models.pdebench_loss import get_loss
from al4pde.modules.fno import FNO1dFullTrajectory, FNO2dFullTrajectory, FNO3dFullTrajectory
from al4pde.prob_models.ensemble import Ensemble


class DummySim:
    autonomous = True

    def get_time(self, t_idx):
        return t_idx


class DummyTask:
    sim = DummySim()


def make_wrapper(model, traj_len):
    wrapper = FNOFullTrajectoryWrapper.__new__(FNOFullTrajectoryWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    wrapper.task = DummyTask()
    wrapper.norm_mode = None
    wrapper.traj_len = traj_len
    wrapper.if_conditional = False
    return wrapper


class IdentityLoss(nn.Module):
    def forward(self, pred, target, reduction=None, initial_step=0):
        return (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2


class DummyFullTrajectoryModel(nn.Module):
    predicts_full_trajectory = True
    initial_step = 2
    loss = IdentityLoss()

    def __init__(self, pred):
        super().__init__()
        self.pred = pred
        self.num_forward_calls = 0

    def forward(self, xx, grid, pde_param=None, t_idx=None):
        self.num_forward_calls += 1
        return self.pred


def make_direct_ensemble(base_models):
    ensemble = Ensemble.__new__(Ensemble)
    nn.Module.__init__(ensemble)
    ensemble.base_models = nn.ModuleList(base_models)
    ensemble.unc_roll_out_mode = "direct"
    ensemble.adapt_unc_to_loss = False
    ensemble.first_model_prediction = False
    ensemble.loss = IdentityLoss()
    return ensemble


class FNOFullTrajectoryTest(unittest.TestCase):
    def test_1d_full_trajectory_wrapper_shape(self):
        model = FNO1dFullTrajectory(num_channels=1, modes1=3, modes2=3, width=8, initial_step=2)
        wrapper = make_wrapper(model, traj_len=5)

        pred = wrapper(
            torch.randn(2, 8, 2, 1),
            torch.randn(2, 8, 1),
        )

        self.assertEqual(pred.shape, (2, 8, 5, 1))

    def test_2d_full_trajectory_wrapper_shape(self):
        model = FNO2dFullTrajectory(num_channels=1, modes1=3, modes2=3, modes3=3, width=8, initial_step=2)
        wrapper = make_wrapper(model, traj_len=5)

        pred = wrapper(
            torch.randn(2, 8, 8, 2, 1),
            torch.randn(2, 8, 8, 2),
        )

        self.assertEqual(pred.shape, (2, 8, 8, 5, 1))

    def test_3d_full_trajectory_wrapper_shape(self):
        model = FNO3dFullTrajectory(num_channels=1, modes1=2, modes2=2, modes3=2, modes4=2, width=4, initial_step=2)
        wrapper = make_wrapper(model, traj_len=3)

        pred = wrapper(
            torch.randn(1, 4, 4, 4, 2, 1),
            torch.randn(1, 4, 4, 4, 3),
        )

        self.assertEqual(pred.shape, (1, 4, 4, 4, 3, 1))

    def test_direct_ensemble_accepts_full_trajectory_fno_wrappers(self):
        wrappers = [
            make_wrapper(FNO1dFullTrajectory(num_channels=1, modes1=3, modes2=3, width=8, initial_step=2), 5),
            make_wrapper(FNO1dFullTrajectory(num_channels=1, modes1=3, modes2=3, width=8, initial_step=2), 5),
        ]
        ensemble = make_direct_ensemble(wrappers)

        mean_traj, unc_traj = ensemble.unc_roll_out(
            torch.randn(2, 8, 2, 1),
            torch.randn(2, 8, 1),
            final_step=5,
        )

        self.assertEqual(mean_traj.shape, (2, 8, 5, 1))
        self.assertEqual(unc_traj.shape, (2, 8, 5, 1))

    def test_training_loss_calls_full_trajectory_model_once(self):
        yy = torch.randn(2, 8, 5, 1)
        model = DummyFullTrajectoryModel(yy + 1)

        pred, loss = get_loss(
            torch.randn(2, 8, 2, 1),
            yy,
            torch.randn(2, 8, 1),
            None,
            torch.zeros(2),
            model,
            current_epoch=0,
            num_epoch=1,
            training_type="full_trajectory",
            noise_std=0.0,
            block_grad=False,
        )

        self.assertEqual(model.num_forward_calls, 1)
        self.assertTrue(torch.equal(pred, yy + 1))
        self.assertEqual(loss.shape, (2, 8, 3, 1))


if __name__ == "__main__":
    unittest.main()
