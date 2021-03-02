import torch
import torch.nn.functional as F

VMAX = 11
VMIN = -6
N_ATOMS = 51

def value_loss(state_values, next_state_values, reward, done, gamma, actual_len):
    expected_state_values = (next_state_values * torch.pow(gamma, actual_len.float()) * (1 - done)) + reward
    return F.mse_loss(state_values, expected_state_values)

def value_loss_with_IS(state_values, next_state_values, new_log_prob, old_log_prob, reward, done, gamma, actual_len):
    # TODO update to V-trace version
    expected_state_values = (next_state_values * torch.pow(gamma, actual_len.float()) * (1 - done)) + reward
    with torch.no_grad():
        truncated_ratio_log = torch.clamp(new_log_prob - old_log_prob, max=0)
        importance_sample_fix = torch.exp(truncated_ratio_log)
    
    value_loss = (F.mse_loss(expected_state_values, state_values, reduction="none") * importance_sample_fix).mean()
    return value_loss

def value_loss_distributional(state_values, next_state_values, new_log_prob, old_log_prob, reward, done, gamma, actual_len):
    batch_size = state_values.shape[0]
    with torch.no_grad():
        support = torch.linspace(VMIN, VMAX, N_ATOMS, device=state_values.device)
        delta_z = (VMAX - VMIN) / (N_ATOMS - 1)

        Tz = reward.unsqueeze(1) + ((1 - done) * (gamma ** actual_len)).unsqueeze(1) * support.unsqueeze(0)  
        Tz.clamp_(min=VMIN, max=VMAX)
        b = (Tz - VMIN) / delta_z
        l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
        l[(u > 0) * (l == u)] -= 1
        u[(l < (N_ATOMS - 1)) * (l == u)] += 1

    m = torch.zeros_like(state_values)
    offset = torch.linspace(0, ((batch_size - 1) * N_ATOMS), batch_size).unsqueeze(1).expand(batch_size, N_ATOMS)\
            .to(state_values.device, torch.int64)
    m.view(-1).index_add_(0, (l + offset).view(-1), (torch.exp(next_state_values) * (u.float() - b)).view(-1))
    m.view(-1).index_add_(0, (u + offset).view(-1), (torch.exp(next_state_values) * (b - l.float())).view(-1))

    value_loss = -torch.sum(m * state_values, dim=1).mean()  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
    return value_loss



def policy_loss(gae, new_log_prob, old_log_prob, clip_eps):
    unclipped_ratio = torch.exp(new_log_prob - old_log_prob)
    clipped_ratio = torch.clamp(unclipped_ratio, 1 - clip_eps, 1 + clip_eps)
    actor_loss = -torch.min(clipped_ratio * gae, unclipped_ratio * gae).mean()
    return actor_loss
