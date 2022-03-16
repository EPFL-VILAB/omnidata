import torch.nn.functional as F


def symmetric_kl_div(logits_P, logits_Q, reduction='batchmean'):
    log_P = F.log_softmax(logits_P, dim=1)
    log_Q = F.log_softmax(logits_Q, dim=1)
    kl_PQ = F.kl_div(log_P, log_Q, log_target=True, reduction=reduction)
    kl_QP = F.kl_div(log_Q, log_P, log_target=True, reduction=reduction)
    return kl_PQ + kl_QP

def jensen_shannon_div(logits_P, logits_Q, reduction='batchmean'):
    log_P = F.log_softmax(logits_P, dim=1)
    log_Q = F.log_softmax(logits_Q, dim=1)
    M = 0.5 * (log_P.exp() + log_Q.exp())
    kl_PM = F.kl_div(log_P, M, log_target=False, reduction=reduction)
    kl_QM = F.kl_div(log_Q, M, log_target=False, reduction=reduction)
    return 0.5 * (kl_PM + kl_QM)

