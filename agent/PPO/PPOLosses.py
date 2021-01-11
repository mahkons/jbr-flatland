import torch
import torch.nn.functional as F

def value_loss(state_values, next_state_values, reward, done, gamma, actual_len):
    expected_state_values = (next_state_values * torch.pow(gamma, actual_len.float()) * (1 - done)) + reward
    return F.mse_loss(state_values, expected_state_values)

def value_loss_with_IS(state_values, next_state_values, new_log_prob, old_log_prob, reward, done, gamma, actual_len):
    expected_state_values = (next_state_values * torch.pow(gamma, actual_len.float()) * (1 - done)) + reward
    with torch.no_grad():
        truncated_ratio_log = torch.clamp(new_log_prob - old_log_prob, max=0)
        importance_sample_fix = torch.exp(truncated_ratio_log)
    
    value_loss = (F.mse_loss(expected_state_values, state_values, reduction="none") * importance_sample_fix).mean()
    return value_loss

def policy_loss(gae, new_log_prob, old_log_prob, clip_eps):
    unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(unclipped_ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
    return actor_loss
