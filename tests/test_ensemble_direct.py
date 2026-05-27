import torch
import torch.nn as nn
import unittest

from al4pde.prob_models.ensemble import Ensemble


class DummyDirectModel(nn.Module):
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.num_forward_calls = 0

    def forward(self, xx, grid, pde_param=None, t_idx=None, return_features=False):
        self.num_forward_calls += 1
        if return_features:
            return self.out, self.out
        return self.out


class IdentityLoss(nn.Module):
    def forward(self, pred, target, reduction=None, initial_step=0):
        assert reduction is None
        return (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2


def make_direct_ensemble(outputs, adapt_unc_to_loss=False, first_model_prediction=False):
    ensemble = Ensemble.__new__(Ensemble)
    nn.Module.__init__(ensemble)
    ensemble.base_models = nn.ModuleList([DummyDirectModel(out) for out in outputs])
    ensemble.unc_roll_out_mode = "direct"
    ensemble.adapt_unc_to_loss = adapt_unc_to_loss
    ensemble.first_model_prediction = first_model_prediction
    ensemble.loss = IdentityLoss()
    return ensemble


class EnsembleDirectTest(unittest.TestCase):
    def test_unc_direct_trajectory_uses_one_forward_per_member_and_variance(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_direct_ensemble(outputs)

        mean_traj, unc_traj = ensemble.unc_direct_trajectory(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
        )

        traj = torch.stack(outputs, dim=0)
        self.assertEqual([m.num_forward_calls for m in ensemble.base_models], [1, 1])
        self.assertEqual(mean_traj.shape, (2, 3, 4, 1))
        self.assertEqual(unc_traj.shape, (2, 3, 4, 1))
        self.assertTrue(torch.equal(mean_traj, traj.mean(dim=0)))
        self.assertTrue(torch.equal(unc_traj, traj.var(dim=0)))

    def test_unc_direct_trajectory_can_use_first_model_prediction(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_direct_ensemble(outputs, first_model_prediction=True)

        mean_traj, _ = ensemble.unc_direct_trajectory(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
        )

        self.assertTrue(torch.equal(mean_traj, outputs[0]))

    def test_unc_roll_out_dispatches_to_direct_mode(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_direct_ensemble(outputs)

        mean_traj, unc_traj = ensemble.unc_roll_out(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
        )

        traj = torch.stack(outputs, dim=0)
        self.assertTrue(torch.equal(mean_traj, traj.mean(dim=0)))
        self.assertTrue(torch.equal(unc_traj, traj.var(dim=0)))

    def test_unc_direct_trajectory_crops_to_final_step(self):
        outputs = [
            torch.arange(10, dtype=torch.float32).reshape(1, 2, 5, 1),
            torch.arange(10, 20, dtype=torch.float32).reshape(1, 2, 5, 1),
        ]
        ensemble = make_direct_ensemble(outputs)

        mean_traj, unc_traj = ensemble.unc_direct_trajectory(
            torch.zeros(1, 2, 1, 1),
            torch.zeros(1, 2, 1),
            final_step=3,
        )

        traj = torch.stack([out[..., :3, :] for out in outputs], dim=0)
        self.assertEqual(mean_traj.shape, (1, 2, 3, 1))
        self.assertEqual(unc_traj.shape, (1, 2, 3, 1))
        self.assertTrue(torch.equal(mean_traj, traj.mean(dim=0)))
        self.assertTrue(torch.equal(unc_traj, traj.var(dim=0)))

    def test_unc_direct_trajectory_can_adapt_uncertainty_to_loss(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_direct_ensemble(outputs, adapt_unc_to_loss=True)

        mean_traj, unc_traj = ensemble.unc_direct_trajectory(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
        )

        self.assertTrue(torch.equal(mean_traj, torch.ones(2, 3, 4, 1) * 2))
        self.assertTrue(torch.equal(unc_traj, torch.ones(2, 3, 4, 1)))

    def test_unc_direct_trajectory_returns_concatenated_features(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_direct_ensemble(outputs)

        mean_traj, unc_traj, features = ensemble.unc_direct_trajectory(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
            return_features=True,
        )

        traj = torch.stack(outputs, dim=0)
        self.assertTrue(torch.equal(mean_traj, traj.mean(dim=0)))
        self.assertTrue(torch.equal(unc_traj, traj.var(dim=0)))
        self.assertTrue(torch.equal(features, torch.concat(outputs, dim=-1)))


if __name__ == "__main__":
    unittest.main()
