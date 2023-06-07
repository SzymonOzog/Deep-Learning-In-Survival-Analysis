import torch 

def negative_likelihood_loss(y_pred, events, eps=1e-8):
    y_pred = y_pred.squeeze()
    hazard_ratio = torch.exp(y_pred)
    risk = torch.cumsum(hazard_ratio, dim=0)
    log_risk = torch.log(risk)
    uncensored_likelihood = y_pred - log_risk
    censored_likelihood = uncensored_likelihood * events
    cum_loss = -torch.sum(censored_likelihood)
    total_events = torch.sum(events) + eps
    return cum_loss / total_events


def deep_hit_loss(y_pred, mask, eps=1e-8):
    reduced = torch.sum(y_pred * mask, dim=1)
    l = reduced.add(eps).log()
    return -l.mean() 
