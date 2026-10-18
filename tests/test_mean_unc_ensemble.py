import unittest
from unittest.mock import ANY, patch

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.prob_models.mean_unc_ensemble import MeanUncEnsemble, build_mean_unc_ensemble
from al4pde.prob_models.mean_unc_internal import MeanUnc


class DummyTask:
    initial_step = 1
    reduced_resolution = 1
    reduced_resolution_t = 1
    skip_initial_steps = 0
    reduced_batch = 1
    img_save_path = "/tmp"
    spatial_dim = 1
    num_channels = 1
    num_pde_param = 0


class DummyLoss(nn.Module):
    def forward(self, pred, target, reduction=None, initial_step=0):
        if reduction is not None:
            raise ValueError(reduction)
        return (pred[..., initial_step:, :] - target[..., initial_step:, :]) ** 2


class DummyMeanUncMember(nn.Module):
    """Minimal internal MeanUnc member exposing only a mean trajectory rollout."""

    predicts_full_trajectory = True

    def __init__(self, trajectory):
        super().__init__()
        self.register_buffer("trajectory", trajectory)
        self.roll_out_calls = 0
        self.forward_calls = 0
        self.init_states = []
        self.training_type = "full_trajectory"
        self.t_train = trajectory.shape[-2]
        self.batch_size = trajectory.shape[0]
        self.val_period = 100
        self.vis_period = 100
        self.loss = DummyLoss()
        self.task_norm = None

    def forward(self, *args, **kwargs):
        self.forward_calls += 1
        raise AssertionError("Direct MeanUncEnsemble should use mean-only roll_out.")

    def roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None, return_features=False):
        self.roll_out_calls += 1
        if return_features:
            raise AssertionError("This test only covers return_features=False.")
        return self.trajectory[..., :final_step, :]

    def init_training(self, al_iter, load_train_data=True, warm_start_model_state=None):
        self.init_states.append(warm_start_model_state)

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        pass

    def validate(self, step, prefix=""):
        pass

    def visualize(self, step):
        pass


class DummyFullTrajectoryWrapper(nn.Module):
    predicts_full_trajectory = True


def make_ensemble(outputs):
    ensemble = MeanUncEnsemble.__new__(MeanUncEnsemble)
    nn.Module.__init__(ensemble)
    ensemble.base_models = nn.ModuleList([DummyMeanUncMember(out) for out in outputs])
    ensemble.unc_roll_out_mode = "direct"
    ensemble.adapt_unc_to_loss = False
    ensemble.first_model_prediction = False
    ensemble.loss = DummyLoss()
    ensemble.synchronized_training = False
    ensemble.val_period = 100
    ensemble.vis_period = 100
    return ensemble


class MeanUncEnsembleTest(unittest.TestCase):
    def test_mean_unc_forwards_full_trajectory_capability_from_wrapper(self):
        member = MeanUnc.__new__(MeanUnc)
        nn.Module.__init__(member)
        member.model_wrapper = DummyFullTrajectoryWrapper()

        self.assertTrue(member.predicts_full_trajectory)

    def test_direct_rollout_uses_member_means_and_ensemble_variance(self):
        outputs = [
            torch.ones(2, 3, 4, 1),
            torch.ones(2, 3, 4, 1) * 3,
        ]
        ensemble = make_ensemble(outputs)

        mean, unc = ensemble.unc_roll_out(
            torch.zeros(2, 3, 1, 1),
            torch.zeros(2, 3, 1),
            final_step=4,
        )

        expected = torch.stack(outputs)
        self.assertTrue(torch.equal(mean, expected.mean(dim=0)))
        self.assertTrue(torch.equal(unc, expected.var(dim=0)))
        self.assertEqual([m.roll_out_calls for m in ensemble.base_models], [1, 1])
        self.assertEqual([m.forward_calls for m in ensemble.base_models], [0, 0])

    def test_non_synchronized_training_passes_one_warm_state_to_each_member(self):
        ensemble = make_ensemble([torch.zeros(1, 2, 2, 1) for _ in range(2)])
        states = [{"weight": torch.tensor(1.0)}, {"weight": torch.tensor(2.0)}]

        ensemble.train_n_epoch(
            al_iter=1,
            num_epoch=0,
            step_offset=0,
            vis=False,
            warm_start_model_states=states,
        )

        self.assertIs(ensemble.base_models[0].init_states[0], states[0])
        self.assertIs(ensemble.base_models[1].init_states[0], states[1])

    def test_non_final_training_validates_outer_ensemble_with_distinct_prefix(self):
        ensemble = make_ensemble([torch.zeros(1, 2, 2, 1) for _ in range(2)])
        with patch.object(ensemble, "validate") as validate:
            ensemble.train_n_epoch(
                al_iter=0,
                num_epoch=1,
                step_offset=10,
                vis=False,
            )

        validate.assert_called_once_with(10, prefix="ensemble/")

    def test_final_iteration_skips_outer_ensemble_validation(self):
        ensemble = make_ensemble([torch.zeros(1, 2, 2, 1) for _ in range(2)])
        with patch.object(ensemble, "validate") as validate:
            ensemble.train_n_epoch(
                al_iter=0,
                num_epoch=1,
                step_offset=10,
                vis=False,
                is_last=True,
            )

        validate.assert_not_called()

    def test_checkpoint_state_dict_includes_each_member(self):
        source_outputs = [
            torch.ones(1, 2, 2, 1),
            torch.ones(1, 2, 2, 1) * 3,
        ]
        source = make_ensemble(source_outputs)
        restored = make_ensemble([torch.zeros_like(out) for out in source_outputs])

        restored.load_state_dict(source.state_dict())

        for member, expected in zip(restored.base_models, source_outputs):
            self.assertTrue(torch.equal(member.trajectory, expected))

    def test_builder_creates_named_mean_unc_members(self):
        cfg = OmegaConf.create({
            "num_models_ensemble": 2,
            "unc_roll_out_mode": "direct",
            "synchronized_training": False,
            "adapt_unc_to_loss": False,
            "first_model_prediction": False,
        })
        members = [
            DummyMeanUncMember(torch.zeros(1, 2, 2, 1)),
            DummyMeanUncMember(torch.ones(1, 2, 2, 1)),
        ]

        with patch(
            "al4pde.prob_models.mean_unc_ensemble.build_mean_unc",
            side_effect=members,
        ) as build_member:
            ensemble = build_mean_unc_ensemble(DummyTask(), cfg)

        self.assertIsInstance(ensemble, MeanUncEnsemble)
        self.assertEqual(list(ensemble.base_models), members)
        self.assertEqual(
            [call.kwargs["name"] for call in build_member.call_args_list],
            ["ens_member_0", "ens_member_1"],
        )

    def test_build_prob_model_dispatches_mean_unc_ensemble_target(self):
        cfg = OmegaConf.create({
            "_target_": "al4pde.prob_models.mean_unc_ensemble.MeanUncEnsemble",
        })
        sentinel = object()
        with patch(
            "al4pde.prob_models.build_prob_model.build_mean_unc_ensemble",
            return_value=sentinel,
        ) as builder:
            result = build_prob_model(DummyTask(), cfg)

        self.assertIs(result, sentinel)
        builder.assert_called_once_with(ANY, cfg)


if __name__ == "__main__":
    unittest.main()
