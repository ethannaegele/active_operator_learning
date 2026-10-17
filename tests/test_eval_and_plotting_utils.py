import json
import os
import tempfile
import unittest

import matplotlib
matplotlib.use("Agg")
import numpy as np

from al4pde.eval_and_plotting_utils import al_comparison_plot_multi_seed


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
            self.assertTrue(os.path.isfile(
                os.path.join(out_dir, "Method A_Method B_RMSE_multi_seed_comparison_plot.png")
            ))

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


if __name__ == "__main__":
    unittest.main()
