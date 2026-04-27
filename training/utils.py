import torch
import matplotlib.pyplot as plt


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_optimizer(model, config):
    name = config.get("optimizer", "sgd")
    lr = config.get("lr", 0.1)
    weight_decay = config.get("weight_decay", 0.0)
    

    if name == "sgd":
        
        nesterov = config.get("nesterov", False)
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=config.get("momentum", 0.9),
            weight_decay=weight_decay,
            nesterov = nesterov
        )

    elif name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

    else:
        raise ValueError(f"Unknown optimizer: {name}")
    
    
def save_checkpoint(path, model,
                    optimizer_h=None, optimizer_p=None,
                    scheduler_h=None, scheduler_p=None,
                    tracker=None,
                    epoch=0, step=0, p_step=0):

    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "step": step,
        "p_step": p_step,
    }

    # ---- optimizers ----
    if optimizer_h is not None:
        ckpt["optimizer_h"] = optimizer_h.state_dict()
    if optimizer_p is not None:
        ckpt["optimizer_p"] = optimizer_p.state_dict()

    # ---- schedulers ----
    if scheduler_h is not None:
        ckpt["scheduler_h"] = scheduler_h.state_dict()
    if scheduler_p is not None:
        ckpt["scheduler_p"] = scheduler_p.state_dict()

    # ---- tracker ----
    if tracker is not None:
        if hasattr(tracker, "state_dict"):
            ckpt["tracker"] = tracker.state_dict()
        else:
            ckpt["tracker"] = tracker.__dict__

    torch.save(ckpt, path)


def load_checkpoint(path, model,
                    optimizer_h=None, optimizer_p=None,
                    scheduler_h=None, scheduler_p=None,
                    tracker=None,
                    map_location="cpu"):

    ckpt = torch.load(path, map_location=map_location)

    # ---- model ----
    model.load_state_dict(ckpt["model"])

    # ---- optimizers ----
    if optimizer_h is not None and "optimizer_h" in ckpt:
        optimizer_h.load_state_dict(ckpt["optimizer_h"])

    if optimizer_p is not None and "optimizer_p" in ckpt:
        optimizer_p.load_state_dict(ckpt["optimizer_p"])

    # backward compatibility (old single optimizer)
    if optimizer_h is not None and "optimizer" in ckpt:
        optimizer_h.load_state_dict(ckpt["optimizer"])

    # ---- schedulers ----
    if scheduler_h is not None and "scheduler_h" in ckpt:
        scheduler_h.load_state_dict(ckpt["scheduler_h"])

    if scheduler_p is not None and "scheduler_p" in ckpt:
        scheduler_p.load_state_dict(ckpt["scheduler_p"])

    # backward compatibility
    if scheduler_h is not None and "scheduler" in ckpt:
        scheduler_h.load_state_dict(ckpt["scheduler"])

    # ---- tracker ----
    if tracker is not None and "tracker" in ckpt:
        if hasattr(tracker, "load_state_dict") and isinstance(ckpt["tracker"], dict):
            try:
                tracker.load_state_dict(ckpt["tracker"])
            except:
                tracker.__dict__.update(ckpt["tracker"])
        else:
            tracker.__dict__.update(ckpt["tracker"])

    epoch = ckpt.get("epoch", 0)
    step = ckpt.get("step", 0)
    p_step = ckpt.get("p_step", 0)

    return epoch, step, p_step

class AccuracyTracker:
    def __init__(self, save_path="accuracy.png"):
        self.save_path = save_path

        # main metrics
        self.train_acc = []
        self.test_acc = []
        self.prior_acc = []
        self.ensemble_acc = []

        # decomposition metrics
        self.x_only_acc = []
        self.z_only_acc = []

    def update(self, train, test, prior, ensemble, x_only=None, z_only=None):
        self.train_acc.append(train)
        self.test_acc.append(test)
        self.prior_acc.append(prior)
        self.ensemble_acc.append(ensemble)

        if x_only is not None:
            self.x_only_acc.append(x_only)
        if z_only is not None:
            self.z_only_acc.append(z_only)

        self.plot()

    def plot(self):
        plt.clf()

        fig, axes = plt.subplots(2, 1, figsize=(8, 10))

        # --- Top plot: main accuracies ---
        axes[0].plot(self.train_acc, label="Train")
        axes[0].plot(self.test_acc, label="Test")
        axes[0].plot(self.prior_acc, label="Prior")
        axes[0].plot(self.ensemble_acc, label="Ensemble")

        axes[0].set_title("Main Accuracy Metrics")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Accuracy")
        axes[0].legend()

        # --- Bottom plot: decomposition ---
        if self.x_only_acc:
            axes[1].plot(self.x_only_acc, label="hypernet(x, 0)")
        if self.z_only_acc:
            axes[1].plot(self.z_only_acc, label="hypernet(0, z)")

        axes[1].set_title("Decomposition Behavior")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(self.save_path)
        plt.close(fig)