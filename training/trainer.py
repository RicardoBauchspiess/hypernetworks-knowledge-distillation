import torch
import torch.nn as nn

import training.losses.ensemble_loss as e_loss
from training.utils import AccuracyTracker

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None,
        device="cuda",
        config=None,
        track = True, # generate graph of accuracy after each epoch
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.config = config or {}

        self.track = track

        self.epochs = self.config.get("epochs", 100)
        self.log_interval = self.config.get("log_interval", 100)

        self.criterion = nn.CrossEntropyLoss()


        self.accuracy_tracker = AccuracyTracker()

    # =====================================================
    def train(self):
        for epoch in range(self.epochs):
            train_acc = self.train_one_epoch(epoch)
            acc_prior, acc_xz, acc_0z, acc_x0, acc_ensemble = self.validate()

            if self.track:
                self.accuracy_tracker.update(
                    train_acc,
                    acc_xz,
                    acc_prior,
                    acc_ensemble,
                    acc_x0,
                    acc_0z
                )
            
    # =====================================================
    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        # don't train predictor during head_start epochs
        if epoch >= self.config.get("head_start",0):
            freeze_predictor = False
        else:
            freeze_predictor = True
            self.model.predictor.eval()

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            if not freeze_predictor:
                self.optimizer["predictor"].zero_grad()
            self.optimizer["hypernet"].zero_grad()

            # Forward predictor to get prior
            prior = self.model.predictor(x)
            if not freeze_predictor:
                self.model.p_iter += 1 # update predictor iteration for dropout scheduling

            # optional processing of prior
            # add here
            
            # forward hypernet based on prior
            out = self.model.hypernet(x, prior.detach()) # detach prior to avoid backprop through predictor
            self.model.h_iter += 1 # update hypernet iteration for dropout scheduling

            loss = self.criterion(out, y)

            if not freeze_predictor:
                loss += self.criterion(prior, y)

            # Backward
            loss.backward()
            if not freeze_predictor:
                self.optimizer["predictor"].step()
            self.optimizer["hypernet"].step()

            # Stats
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)


            # call scheduler after each step
            if self.scheduler is not None:
                if not freeze_predictor:
                    self.scheduler["predictor"].step()
                self.scheduler["hypernet"].step()

        acc = total_correct / total_samples
        loss = total_loss / total_samples

        print(f"[Train] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")

        return acc

    # =====================================================
    def validate(self):
        self.model.eval()

        total_prior = 0
        total_xz = 0
        total_0z = 0
        total_x0 = 0
        total_ensemble = 0
        total_samples = 0


        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)


                prior = self.model.predictor(x)

                no_prior = torch.zeros_like(prior)

                no_x = torch.zeros_like(x)

                output_xz = self.model.hypernet(x, prior)
                output_0z = self.model.hypernet(no_x, prior)
                output_x0 = self.model.hypernet(x, no_prior)

                # compute weighted ensemble logits
                # weights are based on how confident each model is for that sample
                # confidence mode weights based on the probability of the predicted class,
                # margin mode weights based on the difference between the top 2 class probabilities
                # tau > 1 sharpens the weight score, tau=2.0 is often effective
                w = e_loss.compute_ensemble_weights([output_xz, prior], tau = 1, mode = 'confidence')
                ensemble = e_loss.ensemble_logits_from_weights([output_xz, prior], w)

                preds = prior.argmax(dim=1)
                total_prior += (preds == y).sum().item()
                preds = output_xz.argmax(dim=1)
                total_xz += (preds == y).sum().item()
                preds = output_0z.argmax(dim=1)
                total_0z += (preds == y).sum().item()
                preds = output_x0.argmax(dim=1)
                total_x0 += (preds == y).sum().item()
                preds = ensemble.argmax(dim=1)
                total_ensemble += (preds == y).sum().item()

                
                total_samples += x.size(0)

        acc_prior = total_prior / total_samples
        acc_xz = total_xz / total_samples
        acc_0z = total_0z / total_samples
        acc_x0 = total_x0 / total_samples
        acc_ensemble = total_ensemble / total_samples

        print(f"Val Acc: Prior: {acc_prior:.4f}, (x,z): {acc_xz:.4f}, (0,z): {acc_0z:.4f}, (x,0): {acc_x0:.4f}, Ensemble: {acc_ensemble:.4f}")


        return acc_prior, acc_xz, acc_0z, acc_x0, acc_ensemble