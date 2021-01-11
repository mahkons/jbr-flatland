import torch
import os
from torch.distributions import Categorical
from copy import deepcopy

from networks.ActorCritic import ActorNet, CriticNet

class PPOController():
    def __init__(self, params, device):
        super(PPOController, self).__init__()
        self.device = device
        self.params = params

        self.state_sz = params.state_sz
        self.action_sz = params.action_sz
        self.gamma = params.gamma
        self.entropy_coeff = params.entropy_coeff

        self.actor_net = ActorNet(self.state_sz, self.action_sz, params.actor_layers_sz, params.neighbours_depth).to(device)
        self.target_actor = deepcopy(self.actor_net)
        self.critic_net = CriticNet(self.state_sz, self.action_sz, params.critic_layers_sz).to(device)

    def _make_logits(self, states, neighbours_states):
        thoughts = self.actor_net.think(states)

        neighbours_thoughts = torch.zeros(neighbours_states.shape[0], neighbours_states.shape[1], 128, device=self.device)
        valid_neighbours = torch.norm(neighbours_states, dim=2) > 1e-9
        if torch.any(valid_neighbours):
            neighbours_thoughts[valid_neighbours] = self.actor_net.think(neighbours_states[valid_neighbours].view(-1, self.state_sz))
        neighbours_signals = self.actor_net.intent(neighbours_thoughts)
        logits = self.actor_net.act(thoughts, neighbours_signals)
        return logits


    def select_action_batch(self, handles, state_dict, neighbours_state, train=True):
        if not handles:
            return {}, {}
        states = torch.stack([state_dict[handle] for handle in handles])
        neighbours_states = torch.stack([torch.stack(neighbours_state[handle]) for handle in handles])
        with torch.no_grad():
            logits = self._make_logits(states, neighbours_states)
            action_distribution = Categorical(logits=logits)
            if train:
                actions = action_distribution.sample()
            else:
                logits = action_distribution.logits
                actions = torch.argmax(logits, dim=1)
            log_probs = action_distribution.log_prob(actions)
        return dict(zip(handles, actions)), dict(zip(handles, log_probs))


    def fast_select_actions(self, handles, state_dict, neighbours_handles, train=True):
        if not handles:
            return {}
        all_handles = list(set().union(handles, *[neighbours_handles[handle] for handle in handles]))
        with torch.no_grad():
            all_states = torch.stack([state_dict[handle] if handle != -1 else torch.zeros(self.state_sz) for handle in all_handles])
            all_thoughts = self.actor_net.think(all_states)
            for i, h in enumerate(all_handles):
                if h == -1:
                    all_thoughts[i] = torch.zeros(128, dtype=torch.float)
            all_intents = self.actor_net.intent(all_thoughts)
            
            all_thoughts = dict(zip(all_handles, all_thoughts))
            all_intents = dict(zip(all_handles, all_intents))

            thoughts = torch.stack([all_thoughts[handle] for handle in handles])
            neighbours_signals = torch.stack([torch.stack(
                [all_intents[opp_handle] for opp_handle in neighbours_handles[handle]]
                ) for handle in handles])
            logits = self.actor_net.act(thoughts, neighbours_signals)
            action_distribution = Categorical(logits=logits)
            actions = action_distribution.sample()
        return dict(zip(handles, actions))


    def update_net_params(self, net_params):
        actor_state_dict, critic_state_dict, target_actor_state_dict = net_params
        self.actor_net.load_state_dict(actor_state_dict)
        self.critic_net.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(target_actor_state_dict)

    def get_net_params(self, device=None):
        actor_state, critic_state = self.actor_net.state_dict(), self.critic_net.state_dict()
        target_actor_state = self.target_actor.state_dict()
        if device is not None and device != self.device:
            for key, value in actor_state.items():
                actor_state[key] = value.to(device)
            for key, value in critic_state.items():
                critic_state[key] = value.to(device)
            target_actor_state = {k: v.to(device) for k, v in target_actor_state.items()}
        return actor_state, critic_state, target_actor_state

    def hard_update(self):
        actor_state_dict, _, _ = self.get_net_params(self.device)
        self.target_actor.load_state_dict(actor_state_dict)

    def soft_update(self, tau):
        _soft_update_net(self.target_actor, self.actor_net, tau)
    
    def load_controller(self, path):
        model = torch.load(path)
        self.actor_net.load_state_dict(model['actor'])
        self.critic_net.load_state_dict(model['critic'])
        self.hard_update()

    def save_controller(self, dirpath, name="controller.torch"):
        torch.save(self.params, os.path.join(dirpath, "params.torch"))
        actor_state_dict, critic_state_dict, _ = self.get_net_params(device=torch.device("cpu"))
        torch.save({'actor': actor_state_dict, 'critic': critic_state_dict},
                os.path.join(dirpath, name))

def _soft_update_net(target_net, net, tau):
    for target_param, local_param in zip(target_net.parameters(), net.parameters()):
        target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
