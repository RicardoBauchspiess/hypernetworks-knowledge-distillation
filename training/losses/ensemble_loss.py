import torch
import torch.nn.functional as F


def compute_ensemble_weights(
    logits_list,
    targets=None,
    tau=1.0,
    base_w=0.0,
    mode="confidence",  # "confidence" or "margin"
    normalize = False,
    eps=1e-8
):
    """
    logits_list: list of logits OR (logits, weight)
    targets: (B,) or None

    returns:
        weights: (K, B)
    """

    logits_only = []
    model_weights = []

    for item in logits_list:
        if isinstance(item, tuple):
            logit, w = item
        else:
            logit, w = item, 1.0

        logits_only.append(logit)
        model_weights.append(w)

    K = len(logits_only)
    B = logits_only[0].shape[0]
    device = logits_only[0].device

    scores = []

    for logit, m_w in zip(logits_only, model_weights):
        prob = F.softmax(logit.detach(), dim=-1)

        # ---------- CONFIDENCE ----------
        if mode == "confidence":
            # hihger confidence in the correct or top class means the model is 
            # better for that sample
            if targets is not None:
                score = prob[torch.arange(B, device=device), targets]
            else:
                score = prob.max(dim=-1).values

        # ---------- MARGIN ----------
        elif mode == "margin":
            # higher margin between correct and highest wrong class, or between the
            # top 2 classes, means the model is better for that sample
            if targets is not None:
                p_correct = prob[torch.arange(B, device=device), targets]

                prob_clone = prob.clone()
                prob_clone[torch.arange(B, device=device), targets] = 0
                p_wrong_max = prob_clone.max(dim=-1).values

                margin = p_correct - p_wrong_max
                score = (1.0 + margin)/2
            else:
                top2 = prob.topk(2, dim=-1).values
                margin = top2[:, 0] - top2[:, 1]
                score = (1.0 + margin)/2
        else:
            raise ValueError(f"Unknown mode: {mode}")

        score = (score + eps).pow(tau) * m_w
        scores.append(score)

    scores = torch.stack(scores, dim=0)  # (K, B)

    # normalize, make moodels have same scale, useful for simetrical/equivalent models
    # might hinder performance if one model is better than the other
    # don't normalize during testing to keep full advantage of best model
    if normalize:
        # per model normalization
        scores = scores / (scores.mean(dim=1, keepdim=True) + eps)


    # per sample normalization
    weights = scores / (scores.sum(dim=0, keepdim=True) + eps)

        
    # apply base weight
    if base_w > 0:
        uniform = 1/K
        weights = base_w*uniform + (1 - base_w) * weights
    
    
    return weights  # (K, B)

def ensemble_logits_from_weights(logits_list, weights):
    """
    logits_list: list of logits OR (logits, weight)
    weights: (K, B)

    returns:
        (B, C)
    """

    logits_only = [l if not isinstance(l, tuple) else l[0] for l in logits_list]

    ensemble_logits = 0
    for w, logit in zip(weights, logits_only):
        ensemble_logits = ensemble_logits + w.unsqueeze(-1) * logit

    return ensemble_logits