import os
import gc
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from al4pde.models.dataloader import NPYDataset
import hydra
import wandb
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr, t as student_t
from matplotlib.markers import MarkerStyle


from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.utils import load_checkpoint

def setup_loader(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test/val set data, eg data/data/Burgers/val
                            current_directory: str,
                            checkpoint_name: str = '4.pt',
                            config_name: str = 'main',
                            load_eval_model: bool = False,
                            load_sel_model: bool = False
                            ):
    if load_eval_model and load_sel_model:
        raise ValueError("Only load one of eval_model or sel_model during a" \
        "single call to setup_loader.")
    os.chdir(current_directory)
    config_dir = os.path.join(current_directory, "config")
    config_dir = os.path.abspath(config_dir)
    
    with hydra.initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
        cfg = hydra.compose(
            config_name=config_name,
            overrides=overrides,
        )
    root_run_save_path = os.path.join(cfg.task.run_save_path, run_id)
    if load_eval_model:
        checkpoint_run_save_path = os.path.join(root_run_save_path, 'eval_model')
    elif load_sel_model:
        checkpoint_run_save_path = os.path.join(root_run_save_path, 'sel_model')
    else:
        checkpoint_run_save_path = root_run_save_path

    task = hydra.utils.instantiate(cfg.task, run_save_path=root_run_save_path)
    model_cfg = cfg.sel_model if load_sel_model else cfg.prob_model
    prob_model = build_prob_model(task, model_cfg)

    al_iter, sampling_finished = load_checkpoint(
        checkpoint_run_save_path,
        checkpoint_name,
        prob_model,
        train=False,
        load_train_data=True,
    )

    # This should point to the test set folder, not an individual file.
    #test_folder = "data/data/Burgers/test"

    test_data = NPYDataset(
        pde_name=task.pde_name,
        folders=[test_folder],
        reduced_resolution=prob_model.reduced_resolution,
        reduced_resolution_t=prob_model.reduced_resolution_t,
        reduced_batch=prob_model.reduced_batch,
        initial_step=prob_model.initial_step,
        skip_initial_steps=prob_model.skip_initial_steps,
        save_in_reduced_res=task.save_in_reduced_res,
    ).set_num_steps(None)

    return prob_model, test_data, task, al_iter, cfg

def make_json_safe(x, ndigits=8):
    # PyTorch / NumPy scalar
    if hasattr(x, "item"):
        try:
            x = x.item()
        except Exception:
            pass

    # Normal Python / NumPy float-like values
    if isinstance(x, float):
        return round(x, ndigits)

    if isinstance(x, int):
        return x

    if hasattr(x, "__float__"):
        try:
            return round(float(x), ndigits)
        except Exception:
            pass
    # return result in json friendly format
    return str(x)
    
    


def model_error_histogram_plotter(
                                  run_id: str, 
                                  overrides: list[str],
                                  test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                                  out_dir: str, # directory to save plots under
                                  pde_name: str, # eg 'burgers'
                                  time_index: int | list[int] | None, # time index to plot
                                  current_directory: str,
                                  sample_index: int | list[int] = 0, # index of observation from test set to examine
                                  slice_by_time: bool = True, # plot shows errors for a single time slice, or all errors across all time slices?
                                  slice_by_space: bool = False, # plot shows errors for a single spatial point?
                                  x_index: int | list[int] | None = None,
                                  checkpoint_name: str = '4.pt',
                                  config_name: str = 'main',
                                  load_eval_model: bool = False,
                                  show_theoretical_dist: bool = True,
                                  show_plots: bool = True
                                  ):
    if (time_index is not None and not slice_by_time) or (time_index is None and slice_by_time):
        raise ValueError("Invalid combination. If time_index is provided," \
        "slice_by_time must be True. If time_index is not provided," \
        "slice_by_time must be False.")
    if slice_by_space and slice_by_time:
        raise ValueError("slice_by_time and slice_by_space cannot both be True")
    if (x_index is not None and not slice_by_space) or (x_index is None and slice_by_space):
        raise ValueError("Invalid combination. If x_index is provided," \
        "slice_by_space must be True. If x_index is not provided," \
        "slice_by_space must be False.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pde_name = pde_name.lower()
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id, 
                                                  overrides, 
                                                  test_folder,
                                                  current_directory,
                                                  checkpoint_name=checkpoint_name,
                                                  config_name=config_name,
                                                  load_eval_model=load_eval_model)
    if task.spatial_dim != 1:
        raise ValueError("Only 1D tasks currently supported.")

    os.chdir(current_directory)

    # Pick one example. Dataset indexing returns:
    # xx: initial condition window
    # yy: full trajectory segment
    # grid: spatial grid
    # pde_param: PDE parameter
    # t_idx: starting time index
    sample_index_list = [sample_index] if isinstance(sample_index, int) else sample_index
    if slice_by_time:
        if isinstance(time_index, int):
            index_list = [time_index]
        else:
            index_list = time_index
    elif slice_by_space:
        if isinstance(x_index, int):
            index_list = [x_index]
        else:
            index_list = x_index
    else:
        index_list = ["ALL"]
    for sample_idx in sample_index_list:
        xx, yy, grid, pde_param, t_idx = test_data[sample_idx]

        # Add batch dimension and move to device.
        xx = xx.unsqueeze(0).to(device)              # [1, x, initial_step, channels]
        yy = yy.unsqueeze(0).to(device)              # [1, x, time, channels]
        grid = grid.unsqueeze(0).to(device)          # [1, x, spatial_dim]
        pde_param = pde_param.unsqueeze(0).to(device) # [1, num_pde_params]
        t_idx = t_idx.unsqueeze(0).to(device)         # [1]

        prob_model.eval()

        with torch.no_grad():
            # yy.shape[-2] is the trajectory length in time.
            pred, unc = prob_model.unc_roll_out(
                xx=xx,
                grid=grid,
                final_step=yy.shape[-2],
                pde_param=pde_param,
                t_idx=t_idx,
                )

        os.makedirs(out_dir, exist_ok=True)
        for idx in index_list:
            if slice_by_time:
                # errors at a specific time slice
                true_np = yy[0, :, idx, 0].detach().cpu().numpy()
                pred_np = pred[0, :, idx, 0].detach().cpu().numpy()
                if show_theoretical_dist:
                    unc_np = unc[0, 0, idx, 0].detach().cpu().numpy() # should be a scalar, ie a np.float64 or similar, representing the uncertainty for that time slice
                err_np = pred_np - true_np
                err_np = err_np.reshape(-1)
            elif slice_by_space:
                # errors at a specific space slice
                true_np = yy[0, idx, :, 0].detach().cpu().numpy()
                pred_np = pred[0, idx, :, 0].detach().cpu().numpy()
                if show_theoretical_dist:
                    # placeholder behavior since unc model is not trained to predict one unc per spatial point
                    unc_np = unc[0:1, idx:idx+1, :, 0:1].detach.cpu().mean(dim=-2).item() # mean uncertainty at that spatial point across all time
                    unc_np = np.float32(unc_np)
                err_np = pred_np - true_np
                err_np = err_np.reshape(-1)
            else:
                true_np = yy[0, :, :, 0].detach().cpu().numpy()
                pred_np = pred[0, :, :, 0].detach().cpu().numpy()
                if show_theoretical_dist:
                    # placeholder behavior since unc model is not trained to predict one global unc
                    unc_np = unc[0, :, :, :].detach().cpu().numpy().mean() # single scalar uncertainty for all space/time/channels
                err_np = pred_np - true_np
                err_np = err_np.reshape(-1)

            # if show_theoretical_dist:
            #     print(f"uncertainty at sample index {sample_idx}, time index {time_idx}:", unc_np)

            plt.figure(figsize=(6, 4))
            counts, bin_edges, patches = plt.hist(err_np, 
                                                density=True, 
                                                color = 'red', 
                                                edgecolor="black", 
                                                label='Observed Error Distribution')
            if show_theoretical_dist:
                x = np.linspace(bin_edges[0], bin_edges[-1], num = 1000)
                y = 1 / np.sqrt(2 * np.pi * unc_np**2) * np.exp(-x**2 / (2 * unc_np**2))
                plt.plot(x, y, color = 'blue', label='Theoretical Error Distribution')
            if slice_by_time:
                plt.title(f"Model errors at time index {idx}")
            elif slice_by_space:
                plt.title(f"Model errors at x index {idx}")
            else:
                plt.title("Model errors across all space and time")
            plt.xlabel("Error")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            if slice_by_time:
                plt.savefig(f"{out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_time_index_{idx}.png", dpi=200)
            elif slice_by_space:
                plt.savefig(f"{out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_x_index_{idx}.png", dpi=200)
            else:
                plt.savefig(f"{out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_{idx}.png", dpi=200)
            if show_plots:
                plt.show()
            plt.close()

            if slice_by_time:
                print(f"Saved plot to: {out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_time_index_{idx}.png")
            elif slice_by_space:
                print(f"Saved plot to: {out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_x_index_{idx}.png")
            else:
                print(f"Saved plot to: {out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_{idx}.png")


def test_trajectory_visualizer(run_id: str, 
                                overrides: list[str],
                                test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                                out_dir: str, # directory to save plots under
                                pde_name: str, # eg 'burgers'
                                current_directory: str,
                                sample_index: int | list[int] = 0, # index of observation from test set to examine
                                checkpoint_name: str = '4.pt',
                                config_name: str = 'main',
                                load_eval_model: bool = False,
                                show_plots: bool = True
                                ):
    os.chdir(current_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde_name = pde_name.lower()
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                          overrides,
                          test_folder,
                          current_directory,
                          checkpoint_name=checkpoint_name,
                          config_name=config_name,
                          load_eval_model=load_eval_model)
    if task.spatial_dim != 1:
        raise ValueError("Only 1D tasks currently supported.")
    
    prob_model.eval()

    sample_index_list = [sample_index] if isinstance(sample_index, int) else sample_index
    for sample_idx in sample_index_list:
        xx, yy, grid, pde_param, t_idx = test_data[sample_idx]

        # Add batch dimension and move to device.
        xx = xx.unsqueeze(0).to(device)              # [1, x, initial_step, channels]
        yy = yy.unsqueeze(0).to(device)              # [1, x, time, channels]
        grid = grid.unsqueeze(0).to(device)          # [1, x, spatial_dim]
        pde_param = pde_param.unsqueeze(0).to(device) # [1, num_pde_params]
        t_idx = t_idx.unsqueeze(0).to(device)         # [1]

        with torch.no_grad():
        # yy.shape[-2] is the trajectory length in time.
            pred = prob_model.roll_out(
                xx=xx,
                grid=grid,
                final_step=yy.shape[-2],
                pde_param=pde_param,
                t_idx=t_idx,
            )
        
        # print("xx shape:", xx.shape)
        # print("yy shape:", yy.shape)
        # print("pred shape:", pred.shape)
        # print("pde_param:", pde_param)

        rmse = torch.sqrt(torch.mean((pred - yy) ** 2)).item()
        rel_l2 = (
            torch.linalg.norm((pred - yy).reshape(1, -1), dim=1)
            / torch.linalg.norm(yy.reshape(1, -1), dim=1)
        ).item()

        print(f"Sample index {sample_idx} RMSE:", rmse)
        print(f"Sample index {sample_idx} Relative L2:", rel_l2)
        print(device)

        os.makedirs(out_dir, exist_ok=True)

        true_np = yy[0, ..., 0].detach().cpu().numpy()
        pred_np = pred[0, ..., 0].detach().cpu().numpy()
        err_np = abs(pred_np - true_np)

        # print("Prediction:", pred_np)
        # print("True function:", true_np)

        plt.figure(figsize=(6, 4))
        plt.imshow(true_np.T, aspect="auto", origin="lower")
        plt.title(f"True {pde_name} trajectory".title())
        plt.xlabel("x index")
        plt.ylabel("time index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/true_{pde_name}_{run_id}_sample_index_{sample_idx}.png", dpi=200)
        if show_plots:
            plt.show()
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(pred_np.T, aspect="auto", origin="lower")
        plt.title(f"Predicted {pde_name} trajectory".title())
        plt.xlabel("x index")
        plt.ylabel("time index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pred_{pde_name}_{run_id}_sample_index_{sample_idx}.png", dpi=200)
        if show_plots:
            plt.show()
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(err_np.T, aspect="auto", origin="lower")
        plt.title("Absolute Error")
        plt.xlabel("x index")
        plt.ylabel("time index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/err_{pde_name}_{run_id}_sample_index_{sample_idx}.png", dpi=200)
        if show_plots:
            plt.show()
        plt.close()

    print("Saved plots to:", out_dir)


def error_unc_correlation(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                            current_directory: str,
                            checkpoint_name: str = '4.pt',
                            config_name: str = 'main',
                            load_eval_model: bool = False,
                            load_sel_model: bool = False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(current_directory)
    if load_sel_model and not load_eval_model:
        raise ValueError
    if load_eval_model:
        eval_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                    overrides,
                                                    test_folder,
                                                    current_directory,
                                                    checkpoint_name=checkpoint_name,
                                                    config_name=config_name,
                                                    load_eval_model=load_eval_model,
                                                    load_sel_model=False)
        eval_model.eval()
        prob_model = eval_model
    if load_sel_model:
        sel_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                            overrides,
                                                            test_folder,
                                                            current_directory,
                                                            checkpoint_name=checkpoint_name,
                                                            config_name=config_name,
                                                            load_eval_model=False,
                                                            load_sel_model=load_sel_model)
        sel_model.eval()
    if not load_eval_model and not load_sel_model:
        prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                            overrides,
                                                            test_folder,
                                                            current_directory,
                                                            checkpoint_name=checkpoint_name,
                                                            config_name=config_name,
                                                            load_eval_model=load_eval_model,
                                                            load_sel_model=False)
        prob_model.eval()

    test_loader = DataLoader(
                    test_data,
                    batch_size=prob_model.batch_size,
                    shuffle=False
    )

    all_uncs = [None] * len(test_loader)
    all_sq_errs = [None] * len(test_loader)
    for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(test_loader):

        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)
        param = param.to(device)
        t_idx = t_idx.to(device)

        with torch.no_grad():
            if not load_eval_model and not load_sel_model:
                pred, unc = prob_model.unc_roll_out(
                    xx,
                    grid,
                    yy.shape[-2],
                    pde_param=param,
                    t_idx=t_idx,
                )
            elif load_eval_model and load_sel_model:
                pred = eval_model.roll_out(xx, 
                                           grid, 
                                           yy.shape[-2], 
                                           pde_param=param,
                                           t_idx=t_idx)
                _, unc = sel_model.unc_roll_out(xx, 
                                                grid,
                                                yy.shape[-2],
                                                pde_param=param,
                                                t_idx=t_idx)
                prob_model = eval_model
            elif load_eval_model and not load_sel_model:
                pred, unc = eval_model.unc_roll_out(xx,
                                                    grid,
                                                    yy.shape[-2],
                                                    pde_param=param,
                                                    t_idx=t_idx)
                prob_model = eval_model
            else:
                raise ValueError("Invalid combination of loading eval_model and sel_model")
            sq_err = (pred - yy) ** 2

            # # ignore initial known frames
            sq_err = sq_err[..., prob_model.initial_step:, :]
            if load_sel_model:
                if sel_model.initial_step != prob_model.initial_step:
                    raise RuntimeError(f"Unexpected mismatch: " \
                    f"sel_model.initial_step = {sel_model.initial_step}, " \
                    f"eval_model.initial_step = {prob_model.initial_step}")
            unc_eval = unc[..., prob_model.initial_step:, :]


            per_sample_error = sq_err.reshape(sq_err.shape[0], -1).mean(dim=1)
            per_sample_unc = unc_eval.reshape(unc_eval.shape[0], -1).mean(dim=1)
            all_uncs[batch_idx] = per_sample_unc
            all_sq_errs[batch_idx] = per_sample_error

    all_uncs = torch.cat(all_uncs).reshape(-1)
    all_sq_errs = torch.cat(all_sq_errs).reshape(-1)

    # print(all_uncs)
    # print(all_sq_errs)
    pearson = pearsonr(all_uncs.detach().cpu().numpy(), 
                       all_sq_errs.detach().cpu().numpy()).statistic
    spearman = spearmanr(all_uncs.detach().cpu().numpy(), 
                         all_sq_errs.detach().cpu().numpy()).statistic

    print("pearson: ", pearson)
    print("spearman: ", spearman)

    return {'pearson_uncertainty_error_correlation': pearson, 
            'spearman_uncertainty_error_correlation': spearman}


def _error_unc_correlation_from_loaded_model(prob_model, test_loader):
    """Compute normal error/uncertainty correlation without reloading a model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_uncs = []
    all_sq_errs = []
    for xx, yy, grid, param, t_idx in test_loader:
        xx = xx.to(device)
        yy = yy.to(device)
        grid = grid.to(device)
        param = param.to(device)
        t_idx = t_idx.to(device)

        with torch.no_grad():
            pred, unc = prob_model.unc_roll_out(
                xx,
                grid,
                yy.shape[-2],
                pde_param=param,
                t_idx=t_idx,
            )
            sq_err = (pred - yy) ** 2
            sq_err = sq_err[..., prob_model.initial_step:, :]
            unc_eval = unc[..., prob_model.initial_step:, :]
            all_sq_errs.append(sq_err.reshape(sq_err.shape[0], -1).mean(dim=1).cpu())
            all_uncs.append(unc_eval.reshape(unc_eval.shape[0], -1).mean(dim=1).cpu())

    all_uncs = torch.cat(all_uncs).reshape(-1)
    all_sq_errs = torch.cat(all_sq_errs).reshape(-1)
    pearson = pearsonr(all_uncs.numpy(), all_sq_errs.numpy()).statistic
    spearman = spearmanr(all_uncs.numpy(), all_sq_errs.numpy()).statistic
    print("pearson: ", pearson)
    print("spearman: ", spearman)
    return {'pearson_uncertainty_error_correlation': pearson,
            'spearman_uncertainty_error_correlation': spearman}


def _release_eval_memory():
    """Release evaluation-only Python objects and unused CUDA allocations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def save_final_eval_metrics(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                            prefix: str, # eg final_val/ or test/
                            time_step_name: str, # eg final_val_eval
                            file_name: str, # eg "final_val_eval_results"
                            current_directory: str, # eg "/content/drive/MyDrive/projects/active_operator_learning"
                            checkpoint_name: str = '4.pt',
                            config_name: str = 'main',
                            load_eval_model: bool = False,
                            load_sel_model: bool = False
                            ):
    if load_sel_model and not load_eval_model:
        raise ValueError
    
    os.chdir(current_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                overrides,
                                                test_folder,
                                                current_directory,
                                                checkpoint_name=checkpoint_name,
                                                config_name=config_name,
                                                load_eval_model=load_eval_model,
                                                load_sel_model=False)
    test_loader = DataLoader(
                            test_data,
                            batch_size=prob_model.batch_size,
                            shuffle=False
                        )

    prob_model.eval()

    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity if "entity" in cfg.wandb else None,
        id=run_id,
        resume="must",
        name=cfg.wandb.name
    )

    res = prob_model.evaluate(
        step=al_iter,
        loader=test_loader,
        prefix=prefix,
        time_step_name=time_step_name
    )

    if load_sel_model:
        correlation_metrics = error_unc_correlation(run_id,
                                                    overrides,
                                                    test_folder,
                                                    current_directory,
                                                    checkpoint_name=checkpoint_name,
                                                    config_name=config_name,
                                                    load_eval_model=load_eval_model,
                                                    load_sel_model=True)
    else:
        correlation_metrics = _error_unc_correlation_from_loaded_model(prob_model, test_loader)
    correlation_metrics = {prefix + k: make_json_safe(v) for k, v in correlation_metrics.items()}
    wandb.log(correlation_metrics | {prefix + time_step_name: al_iter})

    res_combined = res | correlation_metrics # combine dictionaries
    save_dir = os.path.join(current_directory, "ckpt_eval_results", run_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}.json")

    with open(save_path, "w") as f:
        json.dump(res_combined, f, indent=2)

    wandb.save(save_path, base_path=save_dir)

    eval_meta = {"checkpoint_name": checkpoint_name,
                 "test_folder": test_folder,
                 "prefix": prefix,
                 "overrides": overrides}
    save_path_meta = os.path.join(save_dir, f"{file_name}_metadata.json")

    with open(save_path_meta, "w") as f:
        json.dump(eval_meta, f, indent=2)

    wandb.save(save_path_meta, base_path=save_dir)

    wandb.finish()
    del prob_model, test_data, test_loader, task, cfg
    _release_eval_memory()


def al_comparison_plot(run_ids: list[str], 
                       file_name_factory: str, # string containing the variable {ckpt}, 
                                # which when filled with the variable gives the file name
                       labels: list[str], # labels for the legend
                       num_data: list[int], # size of dataset for each iter,
                                                    # can add in future eval files,
                                                    # but for now insert manually
                       current_directory: str,
                       out_dir: str,
                       title: str,
                       results_folder_name: str = "ckpt_eval_results",
                       prefix: str = "ckpt_val/", # wandb logging prefix used in the json file
                       metric: str = "RMSE", # metric to plot, must exist in each json file
                       ylabel: str | None = None,
                       threshold: float | None = None, # threshold for horizontal dashed line in plot
                       threshold_color: str = "red",
                       threshold_width: float = 2.0,
                       threshold_label: str = "Error Threshold",
                       xticks: list[int] | None = None,
                       xlim: tuple[float] | None = None,
                       ylim: tuple[float] | None = None,
                       #num_evals: int | None = None,
                       ):
    # if num_data is None and num_evals is None:
    #     raise ValueError
    # if num_data is not None:
    #     if num_evals != len(num_data):
    #         print("num_evals should be the same as the length of num_data. Assigning num_evals = len(num_data).")
    os.chdir(current_directory)
    os.makedirs(out_dir, exist_ok=True)
    if num_data is None or len(num_data) == 0:
        raise ValueError
    num_evals = len(num_data)
    if len(run_ids) != len(labels):
        raise ValueError("run_ids and labels should have the same length")
    if "{ckpt}" not in file_name_factory:
        raise ValueError("file_name_factory should contain '{ckpt}' to allow string formatting for different checkpoints")
    
    if not file_name_factory.endswith(".json"):
        raise ValueError("Must include the .json extension in the file name factory.")
    if "/" in metric:
        raise ValueError("Do not include '/' or the logging prefix in the metric argument.")
    metric_dict = dict()

    if ylabel is None:
        ylabel = metric
    
    # get appropriate matplotlib markers for plotting
    markers = [
        marker
        for marker in MarkerStyle.markers
        if isinstance(marker, str) and marker not in {"", " ", 
                                                      "none", "None", 
                                                      ",", "s", ".", "o", 
                                                      "x", "+"}
    ]
    markers = ['o', 's', 'x', '+'] + markers
    markers = markers[:len(labels)]

    for run_id in run_ids:
        metric_dict[run_id] = []
        for n in range(num_evals):
            folder_path = os.path.join(current_directory, 
                                    results_folder_name, 
                                    run_id) # assumes results folder contains subfolders separated by run_id
            file_name = file_name_factory.format(ckpt=n)
            file_path = os.path.join(folder_path, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            current_metric = float(data[prefix + metric]) # e.g. ckpt_val/RMSE
            metric_dict[run_id].append(current_metric)

    run_ids_str = "_".join(run_ids)
    labeled_metric_dict = dict() # label -> list of e.g. RMSE values

    if num_data is not None:
        plt.figure()
        for idx, (run_id, label, marker) in enumerate(zip(run_ids, labels, markers)):
            plt.plot(num_data, metric_dict[run_id], label=label, marker=marker)
            labeled_metric_dict[label] = metric_dict[run_id]
        if threshold is not None:
            plt.axhline(y=threshold, 
                        color=threshold_color, 
                        linestyle="--", 
                        linewidth=threshold_width, 
                        label=threshold_label)
        if xticks is not None:
            plt.xticks(xticks)
        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)
        plt.xlabel("Training Set Size")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{run_ids_str}_{metric}_comparison_plot.png",
                     dpi=200)
        plt.show()
        plt.close()

        print(labeled_metric_dict)
    
    if num_data is None:
        raise ValueError("Not currently implemented for num_data is None. Supply list of dataset sizes manually.")


def al_comparison_plot_multi_seed(
                       run_ids_by_label: dict[str, list[str]],
                       file_name_factory: str,
                       num_data: list[int],
                       current_directory: str,
                       out_dir: str,
                       title: str,
                       results_folder_name: str = "ckpt_eval_results",
                       prefix: str = "ckpt_val/",
                       metric: str = "RMSE",
                       ylabel: str | None = None,
                       confidence_level: float = 0.95,
                       show_confidence_band: bool = True,
                       threshold: float | None = None,
                       threshold_color: str = "red",
                       threshold_width: float = 2.0,
                       threshold_label: str = "Error Threshold",
                       xticks: list[int] | None = None,
                       xlim: tuple[float] | None = None,
                       ylim: tuple[float] | None = None,
                       show_plots: bool = True,
                       ):
    """Plot seed-averaged active-learning metrics with confidence bands.

    ``run_ids_by_label`` maps each legend label to run IDs from independent
    seeds. Each run must contain one JSON file per checkpoint under
    ``results_folder_name/<run_id>`` following ``file_name_factory``. The
    When ``show_confidence_band`` is true, the plot includes a two-sided
    Student-t confidence interval for the mean over the runs associated with
    each label, calculated separately at every dataset size.

    Returns a dictionary containing the per-label seed values, means, and
    confidence-interval bounds so notebooks can inspect the plotted values.
    """
    if not run_ids_by_label:
        raise ValueError("run_ids_by_label must contain at least one label.")
    if num_data is None or len(num_data) == 0:
        raise ValueError("num_data must contain at least one dataset size.")
    if "{ckpt}" not in file_name_factory:
        raise ValueError("file_name_factory must contain '{ckpt}'.")
    if not file_name_factory.endswith(".json"):
        raise ValueError("file_name_factory must include the .json extension.")
    if "/" in metric:
        raise ValueError("Do not include '/' or the logging prefix in metric.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be strictly between 0 and 1.")

    os.makedirs(out_dir, exist_ok=True)
    if ylabel is None:
        ylabel = metric

    num_evals = len(num_data)
    labels = list(run_ids_by_label)
    markers = ['o', 's', 'x', '+']
    markers.extend(
        marker
        for marker in MarkerStyle.markers
        if isinstance(marker, str) and marker not in {"", " ", "none", "None", ",", "s", ".", "o", "x", "+"}
    )

    plot_data = {}
    plt.figure()
    for idx, label in enumerate(labels):
        run_ids = run_ids_by_label[label]
        if not run_ids:
            raise ValueError(f"Label '{label}' has no run IDs.")

        seed_metrics = []
        for run_id in run_ids:
            run_metrics = []
            folder_path = os.path.join(current_directory, results_folder_name, run_id)
            for ckpt in range(num_evals):
                file_path = os.path.join(folder_path, file_name_factory.format(ckpt=ckpt))
                with open(file_path, "r", encoding="utf-8") as file:
                    data = json.load(file)
                run_metrics.append(float(data[prefix + metric]))
            seed_metrics.append(run_metrics)

        seed_metrics = np.asarray(seed_metrics, dtype=float)
        mean = seed_metrics.mean(axis=0)
        if len(run_ids) == 1:
            ci_half_width = np.zeros_like(mean)
        else:
            standard_error = seed_metrics.std(axis=0, ddof=1) / np.sqrt(len(run_ids))
            critical_value = student_t.ppf((1.0 + confidence_level) / 2.0, len(run_ids) - 1)
            ci_half_width = critical_value * standard_error

        lower = mean - ci_half_width
        upper = mean + ci_half_width
        line, = plt.plot(num_data, mean, label=label, marker=markers[idx % len(markers)])
        if show_confidence_band:
            plt.fill_between(num_data, lower, upper, color=line.get_color(), alpha=0.2)
        plot_data[label] = {
            "run_ids": list(run_ids),
            "seed_metrics": seed_metrics,
            "mean": mean,
            "ci_lower": lower,
            "ci_upper": upper,
        }

    if threshold is not None:
        plt.axhline(y=threshold,
                    color=threshold_color,
                    linestyle="--",
                    linewidth=threshold_width,
                    label=threshold_label)
    if xticks is not None:
        plt.xticks(xticks)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)
    plt.xlabel("Training Set Size")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    labels_str = "_".join(labels)
    output_path = os.path.join(out_dir, f"{labels_str}_{metric}_multi_seed_comparison_plot.png")
    plt.savefig(output_path, dpi=200)
    if show_plots:
        plt.show()
    plt.close()

    return plot_data
    

        
