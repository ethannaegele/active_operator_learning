import unittest
from collections import OrderedDict

import torch
import torch.nn as nn

from al4pde.prob_models.ensemble import Ensemble
from al4pde.prob_models.mean_unc_model import MeanUncModel


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


class DummyTask:
    initial_step = 1
    reduced_resolution = 1
    reduced_resolution_t = 1
    skip_initial_steps = 0
    reduced_batch = 1
    img_save_path = "/tmp"
    spatial_dim = 1
    num_channels = 1
    num_pde_param = 1
    pde_param_names = ["nu"]


class DummyLoss(nn.Module):
    def init_training(self, task_norm, initial_step):
        self.task_norm = task_norm
        self.initial_step = initial_step


class DummyMeanWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(self.model.weight, 0.0)
        self.task_norm = None
        self.first_iter_nb = 1
        self.init_states = []

    def init_training(self, al_iter, load_train_data=True, warm_start_model_state=None):
        self.init_states.append(warm_start_model_state)
        if warm_start_model_state is not None:
            self.model.load_state_dict(warm_start_model_state)


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def make_warm_start_ensemble(num_members=2):
    ensemble = Ensemble.__new__(Ensemble)
    nn.Module.__init__(ensemble)
    ensemble.base_models = nn.ModuleList([DummyWarmStartMember() for _ in range(num_members)])
    ensemble.synchronized_training = False
    ensemble.val_period = 1
    ensemble.vis_period = 100
    return ensemble


def make_warm_start_mean_unc_model():
    mean_model = DummyMeanWrapper()

    def unc_model_factory():
        model = nn.Linear(1, 1, bias=False)
        nn.init.constant_(model.weight, 0.0)
        return model

    return MeanUncModel(
        mean_model,
        unc_model_factory,
        noise=0.0,
        noise_mode="constant",
        block_grad=False,
        max_grad_norm=None,
        step_batch=-1,
        optimizer_factory=lambda params: torch.optim.SGD(params, lr=0.1),
        scheduler_factory=DummyScheduler,
        name="mean_unc_test",
        if_conditional=True,
        task=DummyTask(),
        training_type="full_trajectory",
        t_train=1,
        batch_size=1,
        val_period=1,
        vis_period=1,
        loss=DummyLoss(),
    )


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

    def test_mean_unc_init_training_loads_mean_and_uncertainty_states(self):
        model = make_warm_start_mean_unc_model()
        mean_state = OrderedDict([("weight", torch.tensor([[3.0]]))])
        unc_state = OrderedDict([("weight", torch.tensor([[4.0]]))])
        states = {"mean": mean_state, "unc": unc_state}

        model.init_training(al_iter=1, warm_start_model_states=states)

        self.assertIs(model.mean_model.init_states[0], mean_state)
        self.assertTrue(torch.equal(model.mean_model.model.weight.detach(), torch.tensor([[3.0]])))
        self.assertTrue(torch.equal(model.unc_model.weight.detach(), torch.tensor([[4.0]])))

    def test_mean_unc_init_training_allows_no_warm_start_state(self):
        model = make_warm_start_mean_unc_model()

        model.init_training(al_iter=0, warm_start_model_states=None)

        self.assertIsNone(model.mean_model.init_states[0])
        self.assertTrue(torch.equal(model.mean_model.model.weight.detach(), torch.tensor([[0.0]])))
        self.assertTrue(torch.equal(model.unc_model.weight.detach(), torch.tensor([[0.0]])))


if __name__ == "__main__":
    unittest.main()
