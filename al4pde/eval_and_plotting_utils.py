import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from al4pde.models.dataloader import NPYDataset
import hydra
import wandb
import json
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr


from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.utils import load_checkpoint

def setup_loader(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test/val set data, eg data/data/Burgers/val
                            current_directory: str,
                            checkpoint_name: str = '4.pt',
                            ):
    os.chdir(current_directory)
    config_dir = os.path.join(current_directory, "config")
    config_dir = os.path.abspath(config_dir)
    
    with hydra.initialize_config_dir(version_base="1.3.2", config_dir=config_dir):
        cfg = hydra.compose(
            config_name="main",
            overrides=overrides,
        )
    run_save_path = os.path.join(cfg.task.run_save_path, run_id)

    task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
    prob_model = build_prob_model(task, cfg.prob_model)

    al_iter, sampling_finished = load_checkpoint(
        run_save_path,
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
                                  time_index: int | list[int], # time index to plot
                                  current_directory: str,
                                  sample_index: int | list[int] = 0, # index of observation from test set to examine
                                  checkpoint_name: str = '4.pt',
                                  ):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pde_name = pde_name.lower()
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id, 
                                                  overrides, 
                                                  test_folder,
                                                  current_directory,
                                                  checkpoint_name=checkpoint_name)
    if task.spatial_dim != 1:
        raise ValueError("Only 1D tasks currently supported.")

    os.chdir(current_directory)

    # Pick one example. Dataset indexing returns:
    # xx: initial condition window
    # yy: full trajectory segment
    # grid: spatial grid
    # pde_param: PDE parameter
    # t_idx: starting time index
    #sample_idx = 0
    sample_index_list = [sample_index] if isinstance(sample_index, int) else sample_index
    time_index_list = [time_index] if isinstance(time_index, int) else time_index
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

        for time_idx in time_index_list:
            # errors at a specific time slice
            true_np = yy[0, :, time_idx, 0].detach().cpu().numpy()
            pred_np = pred[0, :, time_idx, 0].detach().cpu().numpy()
            unc_np = unc[0, 0, time_idx, 0].detach().cpu().numpy() # should be a scalar, ie a np.float64 or similar, representing the uncertainty for that time slice
            err_np = pred_np - true_np
            err_np = err_np.reshape(-1)

            # print("Prediction:", pred_np)
            # print("True function:", true_np)
            #print("Errors:", err_np)
            print(f"uncertainty at sample index {sample_idx}, time index {time_idx}:", unc_np)

            plt.figure(figsize=(6, 4))
            counts, bin_edges, patches = plt.hist(err_np, 
                                                density=True, 
                                                color = 'red', 
                                                edgecolor="black", 
                                                label='Observed Error Distribution')
            x = np.linspace(bin_edges[0], bin_edges[-1], num = 1000)
            y = 1 / np.sqrt(2 * np.pi * unc_np**2) * np.exp(-x**2 / (2 * unc_np**2))
            plt.plot(x, y, color = 'blue', label='Theoretical Error Distribution')
            plt.title(f"Model errors at time index {time_idx}")
            plt.xlabel("Error")
            plt.ylabel("Density")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_time_index_{time_idx}.png", dpi=200)
            plt.close()

            print(f"Saved plot to: {out_dir}/err_{pde_name}_hist_{run_id}_sample_index_{sample_idx}_time_index_{time_idx}.png")


def test_trajectory_visualizer(run_id: str, 
                                overrides: list[str],
                                test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                                out_dir: str, # directory to save plots under
                                pde_name: str, # eg 'burgers'
                                current_directory: str,
                                sample_index: int | list[int] = 0, # index of observation from test set to examine
                                checkpoint_name: str = '4.pt',
                                ):
    os.chdir(current_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pde_name = pde_name.lower()
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                          overrides,
                          test_folder,
                          current_directory,
                          checkpoint_name=checkpoint_name)
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
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(pred_np.T, aspect="auto", origin="lower")
        plt.title(f"Predicted {pde_name} trajectory".title())
        plt.xlabel("x index")
        plt.ylabel("time index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/pred_{pde_name}_{run_id}_sample_index_{sample_idx}.png", dpi=200)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.imshow(err_np.T, aspect="auto", origin="lower")
        plt.title("Absolute Error")
        plt.xlabel("x index")
        plt.ylabel("time index")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(f"{out_dir}/err_{pde_name}_{run_id}_sample_index_{sample_idx}.png", dpi=200)
        plt.close()

    print("Saved plots to:", out_dir)


def error_unc_correlation(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                            current_directory: str,
                            checkpoint_name: str = '4.pt',):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(current_directory)
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                overrides,
                                                test_folder,
                                                current_directory,
                                                checkpoint_name=checkpoint_name)
    test_loader = DataLoader(
                    test_data,
                    batch_size=prob_model.batch_size,
                    shuffle=False
    )

    prob_model.eval()

    all_uncs = [None] * len(test_loader)
    all_sq_errs = [None] * len(test_loader)
    for batch_idx, (xx, yy, grid, param, t_idx) in enumerate(test_loader):

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

            # # ignore initial known frames
            sq_err = sq_err[..., prob_model.initial_step:, :]
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


def save_final_eval_metrics(run_id: str, 
                            overrides: list[str],
                            test_folder: str, # folder for test or val set data, eg data/data/Burgers/val
                            prefix: str, # eg final_val/ or test/
                            time_step_name: str, # eg final_val_eval
                            file_name: str, # eg "final_val_eval_results"
                            current_directory: str, # eg "/content/drive/MyDrive/projects/active_operator_learning"
                            checkpoint_name: str = '4.pt',
                            ):
    
    os.chdir(current_directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prob_model, test_data, task, al_iter, cfg = setup_loader(run_id,
                                                overrides,
                                                test_folder,
                                                current_directory,
                                                checkpoint_name=checkpoint_name)
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
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    res = prob_model.evaluate(
        step=al_iter,
        loader=test_loader,
        prefix=prefix,
        time_step_name=time_step_name
    )

    correlation_metrics = error_unc_correlation(run_id,
                                                overrides,
                                                test_folder,
                                                current_directory,
                                                checkpoint_name=checkpoint_name)
    correlation_metrics = {prefix + k: make_json_safe(v) for k, v in correlation_metrics.items()}
    wandb.log(correlation_metrics | {prefix + time_step_name: al_iter})

    res_combined = res | correlation_metrics # combine dictionaries
    save_dir = os.path.join(current_directory, "ckpt_eval_results", run_id)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{file_name}.json")

    with open(save_path, "w") as f:
        json.dump(res_combined, f, indent=2)

    wandb.save(f"{file_name}.json")

    wandb.finish()



    

        
