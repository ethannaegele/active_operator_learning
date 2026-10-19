import json
import os
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import Dataset

from al4pde.eval_and_plotting_utils import (
    al_comparison_plot_multi_seed,
    save_final_eval_metrics,
)


class MultiSeedComparisonPlotTest(unittest.TestCase):
    def _write_metrics(self, root, run_id, values):
        run_dir = os.path.join(root, "ckpt_eval_results", run_id)
        os.makedirs(run_dir)
        for checkpoint, value in enumerate(values):
            path = os.path.join(run_dir, f"ckpt_{checkpoint}_metrics.json")
            with open(path, "w", encoding="utf-8") as file:
                json.dump({"ckpt_val/RMSE": value}, file)

    def test_plot_uses_seed_mean_and_t_confidence_interval(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_metrics(root, "method_a_seed_0", [1.0, 2.0])
            self._write_metrics(root, "method_a_seed_1", [3.0, 4.0])
            self._write_metrics(root, "method_b_seed_0", [5.0, 6.0])
            out_dir = os.path.join(root, "plots")

            with patch("al4pde.eval_and_plotting_utils.plt.fill_between") as fill_between:
                result = al_comparison_plot_multi_seed(
                    run_ids_by_label={
                        "Method A": ["method_a_seed_0", "method_a_seed_1"],
                        "Method B": ["method_b_seed_0"],
                    },
                    file_name_factory="ckpt_{ckpt}_metrics.json",
                    num_data=[10, 20],
                    current_directory=root,
                    out_dir=out_dir,
                    title="Test plot",
                    show_plots=False,
                )

            np.testing.assert_allclose(result["Method A"]["mean"], [2.0, 3.0])
            self.assertTrue(np.all(result["Method A"]["ci_lower"] < [2.0, 3.0]))
            self.assertTrue(np.all(result["Method A"]["ci_upper"] > [2.0, 3.0]))
            np.testing.assert_allclose(result["Method B"]["ci_lower"], [5.0, 6.0])
            np.testing.assert_allclose(result["Method B"]["ci_upper"], [5.0, 6.0])
            self.assertEqual(fill_between.call_count, 2)
            self.assertTrue(os.path.isfile(
                os.path.join(out_dir, "Method A_Method B_RMSE_multi_seed_comparison_plot.png")
            ))

    def test_plot_can_hide_confidence_band(self):
        with tempfile.TemporaryDirectory() as root:
            self._write_metrics(root, "method_a_seed_0", [1.0, 2.0])
            self._write_metrics(root, "method_a_seed_1", [3.0, 4.0])

            with patch("al4pde.eval_and_plotting_utils.plt.fill_between") as fill_between:
                result = al_comparison_plot_multi_seed(
                    run_ids_by_label={"Method A": ["method_a_seed_0", "method_a_seed_1"]},
                    file_name_factory="ckpt_{ckpt}_metrics.json",
                    num_data=[10, 20],
                    current_directory=root,
                    out_dir=os.path.join(root, "plots"),
                    title="Test plot",
                    show_confidence_band=False,
                    show_plots=False,
                )

        np.testing.assert_allclose(result["Method A"]["mean"], [2.0, 3.0])
        self.assertTrue(np.all(result["Method A"]["ci_lower"] < [2.0, 3.0]))
        fill_between.assert_not_called()

    def test_plot_rejects_missing_runs_for_a_label(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(ValueError, "has no run IDs"):
                al_comparison_plot_multi_seed(
                    run_ids_by_label={"Method A": []},
                    file_name_factory="ckpt_{ckpt}_metrics.json",
                    num_data=[10],
                    current_directory=root,
                    out_dir=os.path.join(root, "plots"),
                    title="Test plot",
                    show_plots=False,
                )


class TinyTrajectoryDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, idx):
        xx = torch.zeros(2, 1, 1)
        yy = torch.full((2, 3, 1), float(idx + 1))
        grid = torch.zeros(2, 1)
        param = torch.zeros(1)
        t_idx = torch.tensor(0.0)
        return xx, yy, grid, param, t_idx


class TinyEvalModel:
    batch_size = 2
    initial_step = 1

    def eval(self):
        pass

    def evaluate(self, step, loader, prefix, time_step_name):
        return {prefix + "RMSE": 1.0}

    def unc_roll_out(self, xx, grid, final_step, pde_param=None, t_idx=None):
        pred = torch.zeros(xx.shape[0], xx.shape[1], final_step, xx.shape[-1])
        sample_unc = torch.arange(1, xx.shape[0] + 1, dtype=pred.dtype).reshape(-1, 1, 1, 1)
        return pred, sample_unc.expand_as(pred)


class SaveFinalEvalMetricsTest(unittest.TestCase):
    def test_normal_eval_reuses_loaded_model_for_uncertainty_correlation(self):
        self.addCleanup(os.chdir, os.getcwd())
        with tempfile.TemporaryDirectory() as root:
            model = TinyEvalModel()
            cfg = OmegaConf.create({"wandb": {"project": "test", "name": "test"}})
            setup_result = (model, TinyTrajectoryDataset(), SimpleNamespace(), 3, cfg)

            with patch("al4pde.eval_and_plotting_utils.setup_loader", return_value=setup_result) as setup_loader, \
                 patch("al4pde.eval_and_plotting_utils.error_unc_correlation") as reload_correlation, \
                 patch("al4pde.eval_and_plotting_utils._release_eval_memory") as release_memory, \
                 patch("al4pde.eval_and_plotting_utils.wandb.init"), \
                 patch("al4pde.eval_and_plotting_utils.wandb.log"), \
                 patch("al4pde.eval_and_plotting_utils.wandb.save"), \
                 patch("al4pde.eval_and_plotting_utils.wandb.finish"):
                save_final_eval_metrics(
                    run_id="test_run",
                    overrides=[],
                    test_folder="unused",
                    prefix="test/",
                    time_step_name="al_iter",
                    file_name="metrics",
                    current_directory=root,
                    checkpoint_name="3.pt",
                )

            setup_loader.assert_called_once()
            reload_correlation.assert_not_called()
            release_memory.assert_called_once()
            self.assertTrue(os.path.isfile(
                os.path.join(root, "ckpt_eval_results", "test_run", "metrics.json")
            ))


if __name__ == "__main__":
    unittest.main()
