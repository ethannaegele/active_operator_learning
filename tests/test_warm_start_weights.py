import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from al4pde.prob_models.ensemble import Ensemble


class DummyWarmStartMember(nn.Module):
    def __init__(self):
        super().__init__()
        self.init_states = []
        self.train_calls = []
        self.validate_calls = []
        self.visualize_calls = []
        self.task_norm = None

    def init_training(self, al_iter, load_train_data=True, warm_start_model_state=None):
        self.init_states.append(warm_start_model_state)

    def train_single_epoch(self, current_epoch, total_epoch, num_epoch):
        self.train_calls.append((current_epoch, total_epoch, num_epoch))

    def validate(self, step, prefix=""):
        self.validate_calls.append((step, prefix))

    def visualize(self, step):
        self.visualize_calls.append(step)


def make_warm_start_ensemble(num_members=2):
    ensemble = Ensemble.__new__(Ensemble)
    nn.Module.__init__(ensemble)
    ensemble.base_models = nn.ModuleList([DummyWarmStartMember() for _ in range(num_members)])
    ensemble.synchronized_training = False
    ensemble.val_period = 1
    ensemble.vis_period = 100
    return ensemble


class WarmStartWeightsTest(unittest.TestCase):
    def test_non_synchronized_ensemble_passes_warm_start_state_to_each_member(self):
        ensemble = make_warm_start_ensemble(2)
        states = [
            OrderedDict([("weight", torch.tensor([1.0]))]),
            OrderedDict([("weight", torch.tensor([2.0]))]),
        ]

        ensemble.train_n_epoch(
            al_iter=1,
            num_epoch=1,
            step_offset=10,
            warm_start_model_states=states,
            vis=False,
        )

        self.assertIs(ensemble.base_models[0].init_states[0], states[0])
        self.assertIs(ensemble.base_models[1].init_states[0], states[1])
        self.assertEqual(ensemble.base_models[0].train_calls, [(0, 10, 1)])
        self.assertEqual(ensemble.base_models[1].train_calls, [(0, 10, 1)])

    def test_non_synchronized_ensemble_allows_no_warm_start_state(self):
        ensemble = make_warm_start_ensemble(2)

        ensemble.train_n_epoch(
            al_iter=0,
            num_epoch=1,
            step_offset=0,
            warm_start_model_states=None,
            vis=False,
        )

        self.assertIsNone(ensemble.base_models[0].init_states[0])
        self.assertIsNone(ensemble.base_models[1].init_states[0])

    def test_last_iteration_still_warm_starts_and_trains_first_member(self):
        ensemble = make_warm_start_ensemble(2)
        states = [
            OrderedDict([("weight", torch.tensor([1.0]))]),
            OrderedDict([("weight", torch.tensor([2.0]))]),
        ]

        ensemble.train_n_epoch(
            al_iter=2,
            num_epoch=1,
            step_offset=20,
            is_last=True,
            warm_start_model_states=states,
            vis=False,
        )

        self.assertIs(ensemble.base_models[0].init_states[0], states[0])
        self.assertEqual(ensemble.base_models[0].train_calls, [(0, 20, 1)])
        self.assertIs(ensemble.base_models[1].init_states[0], states[1])
        self.assertEqual(ensemble.base_models[1].train_calls, [])


if __name__ == "__main__":
    unittest.main()
