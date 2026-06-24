from al4pde.prob_models.ensemble import build_ensemble
from al4pde.prob_models.mean_unc_model import build_mean_unc_model


def build_prob_model(task, cfg):
    if cfg._target_ == "al4pde.prob_models.ensemble.Ensemble":
        return build_ensemble(task, cfg)
    elif cfg._target_ == "al4pde.prob_models.mean_unc_model.MeanUncModel":
        return build_mean_unc_model(task, cfg)
    else:
        raise ValueError
