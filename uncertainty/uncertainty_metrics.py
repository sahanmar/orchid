import torch


def get_entropy(tensor: torch.Tensor) -> torch.Tensor:
    mask = tensor == 0
    log = torch.log2(tensor)
    log[mask] = 0
    return -torch.sum(log * tensor, dim=1)


def pavpu_metric(
    pred: torch.Tensor, target: torch.Tensor, uncertainty_threshold: float = 0.5
) -> float:
    """
    PAvPU metric is taken from https://arxiv.org/pdf/1811.12709.pdf
    and calculated as (N_ac + N_ic) / (N_ac + N_ic + N_au + N_iu),
    where
        N_ac - # accurate and certain
        N_ic - # inaccurate and certain
        N_au - # accurate and uncertain
        N_ic - # inaccurate and certain

    args:
        pred (torch.Tensor): [words, 1 + n_ants]
        target (torch.Tensor): [words, 1 + n_ants]
        both target and prediction start with a dummy variable which
        states for no coreference
    returns:
        float

    # TODO create metrics config and send thresholds and window here
    """
    criterium = get_entropy(pred)
    threshold = torch.min(criterium) + uncertainty_threshold * (
        torch.max(criterium) - torch.min(criterium)
    )
    certain = criterium < threshold
    uncertain = criterium >= threshold

    pred_ant = torch.argmax(pred, dim=1)
    target_ant = torch.argmax(target, dim=1)
    accurate = pred_ant == target_ant
    inaccurate = pred_ant != target_ant

    acc_certain = accurate * certain
    acc_uncertain = accurate * uncertain
    inacc_certain = inaccurate * certain
    inacc_uncertain = inaccurate * uncertain

    normalizing_const = (
        torch.sum(acc_certain)
        + torch.sum(inacc_certain)
        + torch.sum(acc_uncertain)
        + torch.sum(inacc_uncertain)
    )

    return float(
        (torch.sum(acc_certain) + torch.sum(inacc_uncertain))
        / normalizing_const
    )
