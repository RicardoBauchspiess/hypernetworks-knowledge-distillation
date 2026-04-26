import torch
import torch.nn as nn

from models.ResNets import ResNet20
from models.HyperNetwork import HyperResNet20

class PredictorHyperNet(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()

        self.num_classes = num_classes

        self.predictor = ResNet20(num_classes)
        self.hypernet = HyperResNet20(num_classes)

        # iterations for curriculum
        self.iter = 0
        self.p_iter = 0

        # Dropout
        self.max_dropout = 0
        self.warmup_dropout = 5000

        self.zero_input = False
        self.no_prior = False

    # =====================================================
    def get_dropout_rate(self):
        p = min(1.0, self.iter / self.warmup_dropout)
        hypernet_dropout = self.max_dropout * p

        p = min(1.0, self.p_iter / self.warmup_dropout)
        predictor_dropout = self.max_dropout * p

        return hypernet_dropout, predictor_dropout

    # =====================================================
    def forward(self, x, x_prior=None):
        if x_prior is None:
            x_prior = x

        # Local flags (avoid mutating state)
        no_prior = self.no_prior
        zero_input = self.zero_input

        if no_prior:
            zero_input = False
        elif zero_input:
            no_prior = False

        h_dropout, p_dropout = self.get_dropout_rate()

        # -------------------------
        # Predictor branch
        # -------------------------
        if no_prior:
            prior = torch.zeros(x.size(0), self.num_classes, device=x.device)
            prior_hyper = prior
        else:
            prior = self.predictor(x_prior, dropout = p_dropout)
            prior_hyper = prior.detach()

            if self.training:
                self.p_iter += 1

        # -------------------------
        # HyperNet branch
        # -------------------------
        if zero_input:
            x = torch.zeros_like(x)

        out = self.hypernet(x, prior_hyper, iter=self.iter, dropout = h_dropout)

        if self.training:
            self.iter += 1

        return out, prior
    

