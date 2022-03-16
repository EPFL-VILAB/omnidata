

def compute_grad_norm_losses(losses, model):
    '''
    Balances multiple losses by weighting them inversly proportional
    to their overall gradient contribution.
    
    Args:
        losses: A dictionary of losses.
        model: A PyTorch model.
    Returns:
        A dictionary of loss weights.
    '''
    grad_norms = {}
    model.zero_grad()
    for loss_name, loss in losses.items():
        loss.backward(retain_graph=True)
        grad_sum = sum([w.grad.abs().sum().item() for w in model.parameters() if w.grad is not None])
        num_elem = sum([w.numel() for w in model.parameters() if w.grad is not None])
        grad_norms[loss_name] = grad_sum / num_elem
        model.zero_grad()

    grad_norms_total = sum(grad_norms.values())

    loss_weights = {}
    for loss_name, loss in losses.items():
        weight = (grad_norms_total - grad_norms[loss_name]) / ((len(losses) - 1) * grad_norms_total)
        loss_weights[loss_name] = weight
        
    return loss_weights