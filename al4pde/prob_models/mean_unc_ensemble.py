"""Ensembles of internal mean/uncertainty FNO models."""

from al4pde.prob_models.ensemble import Ensemble
from al4pde.prob_models.mean_unc_internal import build_mean_unc


class MeanUncEnsemble(Ensemble):
    """An ensemble of :class:`MeanUnc` members.

    Members train their internal uncertainty or loss-prediction heads normally.
    In ``direct`` mode, however, the inherited ensemble implementation calls each
    member's mean-only ``roll_out`` and estimates uncertainty from disagreement
    between member trajectories. The internal head outputs are therefore not used
    for acquisition in this model.
    """


def build_mean_unc_ensemble(task, cfg):
    """Build independently trainable internal MeanUnc members for an ensemble."""
    base_models = [
        build_mean_unc(task, cfg, name="ens_member_" + str(i))
        for i in range(cfg.num_models_ensemble)
    ]
    return MeanUncEnsemble(
        task,
        cfg.unc_roll_out_mode,
        base_models,
        cfg.synchronized_training,
        adapt_unc_to_loss=cfg.get("adapt_unc_to_loss", True),
        first_model_prediction=cfg.first_model_prediction,
    )
