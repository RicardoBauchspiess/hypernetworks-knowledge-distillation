import torch
import torch.nn as nn

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
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.config = config or {}

        self.epochs = self.config.get("epochs", 100)
        self.log_interval = self.config.get("log_interval", 100)

        self.criterion = nn.CrossEntropyLoss()

    # =====================================================
    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
            self.validate(epoch)

            if self.scheduler is not None:
                self.scheduler.step()

    # =====================================================
    def train_one_epoch(self, epoch):
        self.model.train()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        for i, (x, y) in enumerate(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            out, prior = self.model(x)

            loss = self.criterion(out, y)

            # Backward
            loss.backward()
            self.optimizer.step()

            # Stats
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            total_correct += (preds == y).sum().item()
            total_samples += x.size(0)

            if i % self.log_interval == 0:
                print(
                    f"Epoch {epoch} | Step {i} | "
                    f"Loss: {loss.item():.4f}"
                )

        acc = total_correct / total_samples
        loss = total_loss / total_samples

        print(f"[Train] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")

    # =====================================================
    def validate(self, epoch):
        self.model.eval()

        total_loss = 0
        total_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in self.val_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                out, _ = self.model(x)

                loss = self.criterion(out, y)

                total_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                total_correct += (preds == y).sum().item()
                total_samples += x.size(0)

        acc = total_correct / total_samples
        loss = total_loss / total_samples

        print(f"[Val] Epoch {epoch} | Loss: {loss:.4f} | Acc: {acc:.4f}")