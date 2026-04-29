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
        self.h_iter = 0
        self.p_iter = 0

        # Dropout
        self.max_dropout = 0
        self.warmup_dropout = 5000

        self.zero_input = False
        self.no_prior = False

    # =====================================================
    def get_dropout_rate(self):
        p = min(1.0, self.h_iter / self.warmup_dropout)
        hypernet_dropout = self.max_dropout * p

        p = min(1.0, self.p_iter / self.warmup_dropout)
        predictor_dropout = self.max_dropout * p

        return hypernet_dropout, predictor_dropout

    # =====================================================
    def forward(self, x, x_prior=None):
        if x_prior is None:
            x_prior = x

        h_dropout, p_dropout = self.get_dropout_rate()

        # -------------------------
        # Predictor branch
        # -------------------------
        prior = self.predictor(x_prior, dropout = p_dropout)
        prior_hyper = prior.detach()

        if self.training:
            self.p_iter += 1

        # -------------------------
        # HyperNet branch
        # -------------------------

        out = self.hypernet(x, prior_hyper, iter=self.h_iter, dropout = h_dropout)

        if self.training:
            self.h_iter += 1

        return out, prior
    

