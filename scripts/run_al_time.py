import shutil
import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from collections import OrderedDict
import wandb
import os
import time
import copy
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = ".10"

import torch
import jax.numpy as jnp
jnp.arange(0, 100)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from jax.lib import xla_bridge
from al4pde.prob_models.build_prob_model import build_prob_model
from al4pde.acquisition.build_selection import build_strategy
from al4pde.utils import save_checkpoint, set_current_seed, load_checkpoint
from al4pde.evaluation.visualization import end_of_al_iter_plots
from scripts.gen_data import generate_data


@hydra.main(version_base="1.3.2", config_path="../config", config_name="main_time")
def main(cfg: DictConfig):

    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        name=cfg.wandb.name,
        config=OmegaConf.to_container(cfg)
    )
    try:
        al_wall_start = time.time()
        if cfg.restore_checkpoint:
            run_id = cfg.checkpoint_id
        else:
            run_id = run.id

        print("run_id", run_id, flush=True)
        print("torch_device", device)
        print("jax_dev", xla_bridge.get_backend().platform, flush=True)
        use_test = cfg.task.use_test
        if use_test:
            print("using test set")

        # init components
        set_current_seed(cfg.seed, 0, sampling_finished=True, task=None, use_test=use_test)
        run_save_path = os.path.join(cfg.task.run_save_path, run_id)
        task = hydra.utils.instantiate(cfg.task, run_save_path=run_save_path)
        acq_strat = build_strategy(task, cfg.acquisition)
        print("acq", acq_strat, flush=True)
        sel_model = build_prob_model(task, cfg.sel_model)
        eval_model = build_prob_model(task, cfg.prob_model)

        def state_dict_to_cpu(state_dict):
            return OrderedDict(
                (k, v.detach().cpu().clone() if torch.is_tensor(v) else copy.deepcopy(v))
                for k, v in state_dict.items()
            )

        def ensemble_member_states(model):
            """Extract warm-start states from plain or internal-MeanUnc members."""
            states = []
            for member in model.base_models:
                if hasattr(member, "model_wrapper"):
                    states.append(state_dict_to_cpu(member.model_wrapper.model.state_dict()))
                else:
                    states.append(state_dict_to_cpu(member.model.state_dict()))
            return states

        if cfg.wandb.name is None:
            run.name = acq_strat.name + "_" + str(cfg.seed)

        num_epoch = cfg.num_epochs
        num_epoch_sel = cfg.num_epochs_sel

        def save_named_checkpoint(run_save_path, model_name, model, al_iter, sampling_finished):
            save_checkpoint(
                os.path.join(run_save_path, model_name),
                model,
                al_iter,
                sampling_finished,
            )

        first_al_iter = 0

        if cfg.restore_checkpoint:

            first_al_iter, sampling_finished = load_checkpoint(os.path.join(run_save_path, "eval_model"), 
                                                                            cfg.checkpoint_file_name, 
                                                                            eval_model,
                                                                            train=False)
            first_al_iter_sel, sampling_finished_sel = load_checkpoint(os.path.join(run_save_path, "sel_model"), 
                                                                       cfg.checkpoint_file_name, 
                                                                       sel_model,
                                                                       train=False)

            assert first_al_iter == first_al_iter_sel
            assert sampling_finished == sampling_finished_sel

            if not sampling_finished:
                set_current_seed(cfg.seed, first_al_iter, False, task=task, use_test=use_test)
                acq_strat.generate(sel_model, first_al_iter, sel_model.train_loader_full_traj)
                save_named_checkpoint(run_save_path, "eval_model", eval_model, first_al_iter, True)
                save_named_checkpoint(run_save_path, "sel_model", sel_model, first_al_iter, True)
                end_of_al_iter_plots(task, eval_model, first_al_iter)
                first_al_iter += 1

        # generate initial data:
        set_current_seed(cfg.seed, -1, sampling_finished=False, task=task, use_test=use_test)
        if "pregen_initial" in cfg and cfg.pregen_initial:
            print("USING PREGENERATED INITIAL DATA - ONLY FOR DEBUGGING")
            assert cfg.seed == 23  # To avoid forgetting regenerating if using multiple seeds

            initial_data_path = os.path.join(task.data_path, task.pde_name, "initial")
            shutil.copytree(initial_data_path, task.traj_save_path, dirs_exist_ok=True)

        else:
            if not cfg.restore_checkpoint:
                generate_data(task, task.traj_save_path, cfg.task.data_gen.num_initial_batches, "init",
                            cfg.task.data_gen.batch_size)
        
        if not cfg.restore_checkpoint or not cfg.warm_start_weights:
            prev_model_states = None
        else:
            prev_model_states = ensemble_member_states(eval_model)
        
        if not cfg.restore_checkpoint or not cfg.warm_start_weights_sel:
            prev_model_states_sel = None
        else:
            prev_model_states_sel = ensemble_member_states(sel_model)


        for al_iter in range(first_al_iter, cfg.num_al_iter):
            al_iter_start = time.time()
            print("\nactive learning iteration " + str(al_iter), flush=True)
            # retrain ensemble model
            set_current_seed(cfg.seed, al_iter, sampling_finished=True, task=task, use_test=use_test)

            eval_model.train_n_epoch(al_iter, 
                                     num_epoch, 
                                     num_epoch * al_iter, 
                                     warm_start_model_states=prev_model_states)

            sel_train_time = sel_model.train_n_epoch(al_iter, 
                                                     num_epoch_sel, 
                                                     num_epoch_sel * al_iter,
                                                    prefix="sel_model_", 
                                                    warm_start_model_states=prev_model_states_sel)
            
            save_named_checkpoint(run_save_path, "eval_model", eval_model, al_iter, False)
            save_named_checkpoint(run_save_path, "sel_model", sel_model, al_iter, False)


            if cfg.warm_start_weights:
                prev_model_states = ensemble_member_states(eval_model)
            
            if cfg.warm_start_weights_sel:
                prev_model_states_sel = ensemble_member_states(sel_model)

            wandb.log({"al/al_iter": al_iter, "al/num_data": len(sel_model.train_loader_full_traj.dataset),
                       "al/train_time": sel_train_time,
                       "al/cumulative_wall_time": time.time() - al_wall_start,
                       "al/iter_wall_time": time.time() - al_iter_start})

            # generate data
            set_current_seed(cfg.seed, al_iter, sampling_finished=False, task=task, use_test=use_test)

            is_last = al_iter == cfg.num_al_iter - 1
            if not is_last:
                acq_strat.generate(sel_model, al_iter, sel_model.train_loader_full_traj)
                save_named_checkpoint(run_save_path, "eval_model", eval_model, al_iter, True)
                save_named_checkpoint(run_save_path, "sel_model", sel_model, al_iter, True)

            end_of_al_iter_plots(task, eval_model, al_iter, last=is_last)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish(exit_code=0)


if __name__ == "__main__":
    main()
    print("Done with ensemble based active learning experiment.", flush=True)
