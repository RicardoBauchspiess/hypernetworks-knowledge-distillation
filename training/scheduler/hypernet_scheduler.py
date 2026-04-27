import torch
from training.scheduler.warmupcosinescheduler import WarmupCosineScheduler

# Generate separate schedulers for predictor and hypernet
def build_hypernet_schedulers(config, optimizers, steps_per_epoch):
    schedulers = {}

    train_cfg = config.train_hyperparameters
    epochs = train_cfg["epochs"]
    head_start = train_cfg.get("head_start", 0)

    # -------------------------
    # Predictor scheduler
    # -------------------------
    pred_opt_cfg = config.predictor_optimizer
    pred_sched_cfg = config.predictor_scheduler

    effective_epochs = max(0, epochs - head_start)
    total_steps = effective_epochs * steps_per_epoch

    pred_scheduler_config = {
        **pred_sched_cfg,
        "base_lr": pred_opt_cfg["lr"],
        "total_steps": total_steps,
    }

    if "warmup_epochs" in pred_sched_cfg:
        pred_scheduler_config["warmup_steps"] = (
            pred_sched_cfg["warmup_epochs"] * steps_per_epoch
        )

    schedulers["predictor"] = build_scheduler(
        optimizers["predictor"],
        pred_scheduler_config
    )

    # -------------------------
    # Hypernet scheduler
    # -------------------------
    hyper_opt_cfg = config.hypernet_optimizer
    hyper_sched_cfg = config.hypernet_scheduler

    total_steps = epochs * steps_per_epoch

    hyper_scheduler_config = {
        **hyper_sched_cfg,
        "base_lr": hyper_opt_cfg["lr"],
        "total_steps": total_steps,
    }

    if "warmup_epochs" in hyper_sched_cfg:
        hyper_scheduler_config["warmup_steps"] = (
            hyper_sched_cfg["warmup_epochs"] * steps_per_epoch
        )

    schedulers["hypernet"] = build_scheduler(
        optimizers["hypernet"],
        hyper_scheduler_config
    )

    return schedulers


def build_scheduler(optimizer, config):
    name = config.get("scheduler", "cosine")

    if name == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=config["warmup_steps"],
            total_steps=config["total_steps"],
            base_lr=config["lr"],
        )

    elif name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"]
        )

    else:
        return None