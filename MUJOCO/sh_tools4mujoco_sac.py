import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RATE_GTrXL.mem_transformer_v2_GTrXL import Model, CustomTransformerEncoder
from gymnasium import spaces
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################
import gym
#import gymnasium as gym
#import pybullet_envs_gymnasium 
##################################

LOG_STD_MIN = -5
LOG_STD_MAX = 2

def mean_padding(tensor, K):
    if len(tensor.shape) == 3:
        num_envs, context, state_dim = tensor.shape
        if context >= K:
            return tensor #[:, :, :K, :] 
        mean_state = tensor[:,0,:].unsqueeze(1)
        pad_tensor = mean_state.expand(num_envs, K - context, state_dim)
        padded_tensor = torch.cat([pad_tensor, tensor], dim=1)
        #print(f"Padded to context {K}")
        return padded_tensor
    
    else:    
        num_envs, batch_size, context, state_dim = tensor.shape
        if context >= K:
            return tensor #[:, :, :K, :] 
        mean_state = tensor[:,:,0,:].unsqueeze(2)
        pad_tensor = mean_state.expand(num_envs, batch_size, K - context, state_dim)
        padded_tensor = torch.cat([pad_tensor, tensor], dim=2)
        #print(f"Padded to context {K}")
        return padded_tensor
    # if len(tensor.shape) == 3:
    #     num_envs, context, state_dim = tensor.shape
    #     if context >= K:
    #         return tensor #[:, :, :K, :] 
    #     mean_state = tensor.mean(dim=1, keepdim=True) 
    #     pad_tensor = mean_state.expand(num_envs, K - context, state_dim)
    #     padded_tensor = torch.cat([pad_tensor, tensor], dim=1)
    #     print(f"Padded to context {K}")
    #     return padded_tensor
    
    # else:    
    #     num_envs, batch_size, context, state_dim = tensor.shape
    #     if context >= K:
    #         return tensor #[:, :, :K, :] 
    #     mean_state = tensor.mean(dim=2, keepdim=True) 
    #     pad_tensor = mean_state.expand(num_envs, batch_size, K - context, state_dim)
    #     padded_tensor = torch.cat([pad_tensor, tensor], dim=2)
    #     print(f"Padded to context {K}")
    #     return padded_tensor




class PartialObservation(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_indices: list):
        super().__init__(env)
        self.obs_indices = np.array(obs_indices, dtype=int)
        obsspace = env.observation_space

        # Векторизованное индексирование вместо списковых включений
        low = obsspace.low[self.obs_indices]
        high = obsspace.high[self.obs_indices]

        self.observation_space = spaces.Box(
            low=low,
            high=high,
            dtype=np.float32,
        )

    def observation(self, observation):
        # Векторизованное индексирование вместо списковых включений
        return observation[self.obs_indices].astype(np.float32)

class GPUObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, device: torch.device):
        super().__init__(env)
        self.device = device  # Устройство (GPU или CPU)

    def reset(self, **kwargs):
        # Получаем начальное наблюдение
        obs = self.env.reset(**kwargs)
        obs_tensor = self._to_tensor(obs).unsqueeze(0)
        #return {'state': obs_tensor}
        return obs_tensor
        
    def step(self, action):
        # Выполняем шаг в среде
        obs, reward, done, info = self.env.step(action)
        
        # Конвертируем наблюдения, вознаграждения и флаги завершения в тензоры
        obs_tensor = self._to_tensor(obs).unsqueeze(0)
        reward_tensor = self._to_tensor(np.array(reward, dtype=np.float32))
        done_tensor = self._to_tensor(np.array(done, dtype=np.bool_))
        
        #return {'state': obs_tensor}, reward_tensor, done_tensor, info
        return obs_tensor, reward_tensor, done_tensor, info

    def _to_tensor(self, obs: np.ndarray):
        if isinstance(obs, np.ndarray):
            if np.issubdtype(obs.dtype, np.bool_):
                tensor = torch.from_numpy(obs).to(torch.bool)
            elif np.issubdtype(obs.dtype, np.floating):
                tensor = torch.from_numpy(obs).float()
            elif np.issubdtype(obs.dtype, np.integer):
                tensor = torch.from_numpy(obs).long()
            else:
                # Для остальных типов данных используем float по умолчанию
                tensor = torch.tensor(obs, dtype=torch.float32)
            return tensor.to(self.device)
        return obs

    def seed(self, seed: int = None):
        # Вызываем seed метода оригинальной среды один раз
        return self.env.seed(seed)


def env_constructor(env_name: str, seed: int = 1, obs_indices: list = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(env_name)
    env.seed(seed)  # Один раз устанавливаем seed здесь
    
    if obs_indices is not None:
        env = PartialObservation(env, obs_indices)
    
    env = GPUObservationWrapper(env, device)
 
    return env, env.observation_space.shape[-1], env.action_space.shape[-1]



class GaussianActor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean = nn.Linear(256, action_dim)
        self.log_std = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state, deterministic=False, with_logprob=True):
        h = self.net(state)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)
        if deterministic:
            z = mean
        else:
            eps = torch.randn_like(mean)
            z = mean + std * eps  # репараметризация
        action = torch.tanh(z)
        action = self.max_action * action
        if not with_logprob:
            return action
        # лог-правдоподобие со «squash» поправкой
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean, log_std

# --- Twin Critic (Q1/Q2) ---
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)

class Trans_Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim, d_model=256, n_heads=2, num_layers=2, dim_feedforward=512, dropout=0.05,
                 wo_ffn=False, norm_first=True, use_gate=False, gate_mode='GRU', mode='Trans'):
        super().__init__()
        self.d_model = d_model
        # Проекция признаков состояния в пространство трансформера
        self.state_fc = nn.Linear(state_dim, d_model)
        # Стек энкодеров
        self.enc_layers = nn.ModuleList([
            CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, mode)
            for _ in range(num_layers)
        ])
        # Ветка действия до склейки
        self.action_fc = nn.Linear(action_dim, d_model)
        # Две головы Q
        self.q1 = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, 1)
        )

    def _encode_state(self, states_seq):
        # Приводим к (N, T, s_d)
        reshape = None
        if states_seq.dim() == 4:
            n_e, bs, T, s_d = states_seq.shape
            x = states_seq.view(-1, T, s_d)
            reshape = (n_e, bs)
        elif states_seq.dim() == 3:
            x = states_seq
        else:
            raise ValueError(f"states_seq must be (bs,T,s) or (n_e,bs,T,s), got {states_seq.shape}")
        x = self.state_fc(x)                 # (N, T, d_model)
        for layer in self.enc_layers:
            x = layer(x)                     # (N, T, d_model)
        last = x[:, -1, :]                   # (N, d_model)
        if reshape is not None:
            n_e, bs = reshape
            last = last.view(n_e, bs, -1)
        return last

    def forward(self, states_seq, action):
        h = self._encode_state(states_seq)
        a = self.action_fc(action)
        sa = torch.cat([h, a], dim=-1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, states_seq, action):
        h = self._encode_state(states_seq)
        a = self.action_fc(action)
        sa = torch.cat([h, a], dim=-1)
        return self.q1(sa)
    
class Trans_Actor(nn.Module):
    """Актор на базе CustomTransformerEncoder: кодирует последовательность состояний,
    берёт последний токен и предсказывает параметры гауссовского распределения действий."""
    def __init__(self, state_dim, action_dim, max_action, d_model=256, n_heads=2, num_layers=2, dim_feedforward=512, dropout=0.05,
                 log_std_min=LOG_STD_MIN, log_std_max=LOG_STD_MAX, wo_ffn=False, norm_first=True, use_gate=False, gate_mode='GRU', mode='Trans'):
        super().__init__()
        self.max_action = max_action
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.state_fc = nn.Linear(state_dim, d_model)
        self.enc_layers = nn.ModuleList([
            CustomTransformerEncoder(d_model, n_heads, dim_feedforward, dropout, wo_ffn, norm_first, use_gate, gate_mode, mode)
            for _ in range(num_layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.mean = nn.Linear(d_model, action_dim)
        self.log_std = nn.Linear(d_model, action_dim)

    def _encode_state(self, states_seq):
        reshape = None
        if states_seq.dim() == 4:
            n_e, bs, T, s_d = states_seq.shape
            x = states_seq.view(-1, T, s_d)
            reshape = (n_e, bs)
        elif states_seq.dim() == 3:
            x = states_seq
        else:
            raise ValueError(f"states_seq must be (bs,T,s) or (n_e,bs,T,s), got {states_seq.shape}")

        x = self.state_fc(x)
        for layer in self.enc_layers:
            x = layer(x)
        last = x[:, -1, :]  # (n_e*bs, d_model) или (bs, d_model)

        if reshape is not None:
            n_e, bs = reshape
            d = last.size(-1)
            if last.numel() == 0:
                # Вернуть корректный пустой тензор нужной формы
                return last.new_empty(n_e, bs, d)
            last = last.contiguous().view(n_e, bs, d)
        return last

    def forward(self, states_seq, deterministic=False, with_logprob=True):
        h = self._encode_state(states_seq)
        h = self.head(h)
        mean = self.mean(h)
        log_std = torch.clamp(self.log_std(h), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        if deterministic:
            z = mean
        else:
            z = mean + std * torch.randn_like(mean)
        action = torch.tanh(z) * self.max_action
        if not with_logprob:
            return action
        # squash correction
        log_prob = Normal(mean, std).log_prob(z) - torch.log(1 - torch.tanh(z).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean, log_std

class SAC(object):
    def __init__(
        self,
        num_envs,
        obs_mode,
        context_length,
        model_config,  # передаётся для трансформера
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=None,              # если None – автотюнинг
        target_entropy=None,     # если None -> -action_dim
        actor_lr=3e-4,
        critic_lr=1e-3,
        alpha_lr=3e-4,
        grad_clip=1e9,
        preload_weights=None,
    ):
        self.num_envs = num_envs
        self.obs_mode = obs_mode
        self.context_length = context_length
        self.discount = discount
        self.tau = tau
        self.max_action = max_action
        self.grad_clip = grad_clip

        if preload_weights == None:
            # MLP-ветка
            self.actor = GaussianActor(state_dim, action_dim, max_action).to(device)
            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

            # Trans-ветка (требует ваш backbone Model + адаптер)
            #from RATE_GTrXL.mem_transformer_v2_GTrXL import Model  # как у вас в исходнике
            #self.trans_backbone = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode=obs_mode).to(device)
            self.trans_actor = Trans_Actor(state_dim, action_dim, max_action).to(device)
            self.trans_actor_target = copy.deepcopy(self.trans_actor).to(device)
            self.trans_actor_optimizer = torch.optim.Adam(self.trans_actor.parameters(), lr=actor_lr)

            self.trans_RB = Trans_RB(num_envs, 30000, context_length, state_dim, action_dim)
            self.trans_critic = Trans_Critic(state_dim, action_dim, d_model=256).to(device)
            self.trans_critic_target = copy.deepcopy(self.trans_critic).to(device)
            self.trans_critic_optimizer = torch.optim.Adam(self.trans_critic.parameters(), lr=critic_lr)
        else:
            print('Found preload weights!')
            print('***DOWNLOADING WEIGHTS***')
            p2tr = preload_weights[0]
            p2tr_tgt = preload_weights[1]
            p2cr = preload_weights[2]
            p2cr_tgt = preload_weights[3]
            
            self.trans_actor = torch.load(p2tr).to(device)
            self.trans_actor_target = torch.load(p2tr_tgt).to(device)
            self.trans_actor_optimizer = torch.optim.Adam(self.trans_actor.parameters(), lr=3e-4)
            
            self.trans_critic = torch.load(p2cr).to(device)
            self.trans_critic_target = torch.load(p2cr_tgt).to(device)
            self.trans_critic_optimizer = torch.optim.Adam(self.trans_critic.parameters(), lr=3e-4)
        
        
        # Температура (энтропия)
        self.target_entropy = -float(action_dim) if target_entropy is None else target_entropy
        if alpha is None:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
            self._fixed_alpha = None
        else:
            self.log_alpha = None
            self._fixed_alpha = torch.tensor([alpha], device=device)

        self.total_it = 0
        self.eval_counter = 0

    # --------- API ---------
    @property
    def alpha(self):
        if self._fixed_alpha is not None:
            return self._fixed_alpha
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, state, deterministic=True):
        # state: (n_e, s_d)
        a = self.actor(state, deterministic=deterministic, with_logprob=False)
        return a.cpu().numpy()

    # --------- MLP SAC update ---------
    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        # next action & log prob
        with torch.no_grad():
            next_action, next_logp, _, _ = self.actor(next_state, deterministic=False, with_logprob=True)
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha * next_logp
            target_q = reward + not_done * self.discount * target_v
        # critic
        cur_q1, cur_q2 = self.critic(state, action)
        critic_loss = F.mse_loss(cur_q1, target_q) + F.mse_loss(cur_q2, target_q)
        self.critic_optimizer.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()
        # actor
        new_action, logp, _, _ = self.actor(state, deterministic=False, with_logprob=True)
        q1_pi = self.critic.Q1(state, new_action)
        actor_loss = (self.alpha * logp - q1_pi).mean()
        self.actor_optimizer.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()
        # alpha (autotune)
        if self.log_alpha is not None:
            alpha_loss = (self.alpha * (-logp - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()
        # soft update
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

    # --------- Transformer SAC (stage 2) ---------
    def stage_2_train(self, batch_size):
        
        self.total_it += 1
        batch = self.new_trans_RB.sample(batch_size) if hasattr(self, 'new_trans_RB') else self.trans_RB.sample(torch.randperm(self.trans_RB.idx))
        if self.obs_mode == 'state':
            states, actions, rewards, dones, next_states = batch
            img_states = img_next_states = None
        else:
            states, actions, rewards, dones, next_states, img_states, img_next_states = batch
        states = states.to(device)
        next_states = next_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        not_dones = (1 - dones).to(device)

        # --- Critic update ---
        with torch.no_grad():
            na, nlogp, _, _ = self.trans_actor(next_states, deterministic=False, with_logprob=True)
            tq1, tq2 = self.trans_critic_target(next_states[:,:,-1,:], na)
            tv = torch.min(tq1, tq2) - self.alpha * nlogp
            target_q = rewards + not_dones * self.discount * tv
        cq1, cq2 = self.trans_critic(states[:,:,-1,:], actions)
        critic_loss = F.mse_loss(cq1, target_q) + F.mse_loss(cq2, target_q)
        self.trans_critic_optimizer.zero_grad(); critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.trans_critic.parameters(), self.grad_clip)
        self.trans_critic_optimizer.step()

        # --- Actor update ---
        pa, logp, _, _ = self.trans_actor(states, deterministic=False, with_logprob=True)
        q1_pi = self.trans_critic.Q1(states[:,:,-1,:], pa)
        actor_loss = (self.alpha * logp - q1_pi).mean()
        self.trans_actor_optimizer.zero_grad(); actor_loss.backward()
        torch.nn.utils.clip_grad_value_(self.trans_actor.parameters(), self.grad_clip)
        self.trans_actor_optimizer.step()

        # --- Alpha ---
        if self.log_alpha is not None:
            alpha_loss = (self.alpha * (-logp - self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad(); alpha_loss.backward(); self.alpha_optimizer.step()

        # --- Soft updates ---
        for p, tp in zip(self.trans_critic.parameters(), self.trans_critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.trans_actor.parameters(), self.trans_actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        
        
    def train_trans_actor(self, batch_size, additional_ascent, additional_bellman=None):
        self.trans_actor.train()
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        bc_losses = []
        ascent_losses = []
        critic_bc_losses = []
        critic_bellman_losses = []
        for chunk in chunks:
            batch = self.trans_RB.sample(chunk)
            
            states = batch[0]       # n_e, b_s, cont, s_d
            next_states = batch[1]  # n_e, b_s, cont, s_d
            targets = batch[2]      # n_e, b_s, a_d
            rewards = batch[3]      # n_e, b_s, 1
            not_dones = batch[4]    # n_e, b_s, 1
            
            preds = self.trans_actor(states, deterministic=True, with_logprob=False)
            bc_loss = nn.MSELoss()(preds, targets)
            self.trans_actor_optimizer.zero_grad()
            bc_losses.append(bc_loss.item())
            bc_loss.backward()
            self.trans_actor_optimizer.step()
            
            if hasattr(self, 'trans_critic'):  #it means that we train trans_critic by BC (and TD-error)
                with torch.no_grad():
                    tgt_q1, tgt_q2 = self.critic(states[:,:,-1,], targets) #ne, bs, 1
                pred_q1, pred_q2 = self.trans_critic(states, targets)               #ne, bs, 1
                critic_bc_loss = nn.MSELoss()(pred_q1, tgt_q1) + nn.MSELoss()(pred_q2, tgt_q2)
                critic_bc_losses.append(critic_bc_loss.cpu().detach().numpy())
                self.trans_critic_optimizer.zero_grad()
                critic_bc_loss.backward()
                self.trans_critic_optimizer.step()
                
                # if additional_bellman is not None:
                #     with torch.no_grad():
                #         noise = (   torch.randn_like(targets) * self.policy_noise  
                #                 ).clamp(-self.noise_clip, self.noise_clip)
                        
                #         if additional_bellman == 'Trans': # means that target value is generated by Trans 
                #             next_action = (self.trans_actor_target(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                #                     ).clamp(-self.max_action, self.max_action)
                #             target_Q1, target_Q2 = self.trans_critic_target(next_states, next_action)
                        
                #         elif additional_bellman == 'MLP': # means that target value is generated by MLP
                #             next_action = (self.actor(next_states[:,:,-1,:]) + noise  			                 #next_action = (n_e, bs, a_d)
                #                     ).clamp(-self.max_action, self.max_action)
                #             target_Q1, target_Q2 = self.critic_target(next_states[:,:,-1,:], next_action)
                        
                #         target_Q = torch.min(target_Q1, target_Q2) 
                #         target_Q = rewards + not_dones * self.discount * target_Q  #(n_e, b_s, 1)+(n_e, b_s, 1)*const*(n_e, b_s, 1)=(n_e, b_s, 1)
                        
                #     current_Q1, current_Q2 = self.trans_critic(states, targets)
                #     bellman_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                #     critic_bellman_losses.append(bellman_loss.cpu().detach().numpy())
                #     self.trans_critic_optimizer.zero_grad()
                #     bellman_loss.backward()
                #     self.trans_critic_optimizer.step()
                    
                    
                    # '''
                    # TO DO: next_states(ne,bs,cont,sd), reward, not_done ||| or I've already done it before ? 
                    # '''
                
                
            if additional_ascent is not None:
                if additional_ascent == 'Trans':      
                    ascent_loss = -self.trans_critic.Q1( states, self.trans_actor(states, deterministic=True, with_logprob=False) ).mean()
                elif additional_ascent == 'MLP':
                    ascent_loss = -self.critic.Q1( states[:,:,-1,], self.trans_actor(states, deterministic=True, with_logprob=False) ).mean()
                ascent_losses.append(ascent_loss.cpu().detach().numpy())
                self.trans_actor_optimizer.zero_grad()
                ascent_loss.backward()
                self.trans_actor_optimizer.step()
                
                if additional_ascent == 'Trans':      
                    self.trans_critic_optimizer.zero_grad()
                elif additional_ascent == 'MLP':
                    self.critic_optimizer.zero_grad()
                    
        self.trans_RB.reset()
        self.experiment.add_scalar('Trans_BC_loss', np.mean(bc_losses), self.eval_counter)
        if additional_ascent is not None:
            self.experiment.add_scalar('Trans_Ascent_loss', np.mean(ascent_losses), self.eval_counter)
        if hasattr(self, 'trans_critic'):
            self.experiment.add_scalar('Critic_BC_loss', np.mean(critic_bc_losses), self.eval_counter)     
        if additional_bellman is not None:
            self.experiment.add_scalar('Critic_Bellman_loss', np.mean(critic_bellman_losses), self.eval_counter)
        self.eval_counter += 1	
        
        for param, target_param in zip(self.trans_actor.parameters(), self.trans_actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        if hasattr(self, 'trans_critic'):    
            for param, target_param in zip(self.trans_critic.parameters(), self.trans_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)    
            
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()  
        self.trans_actor_optimizer.zero_grad()
        self.trans_critic_optimizer.zero_grad()    
    
    def train_trans_critic(self, batch_size, additional_bellman):
        '''
        store this function in order to train critic separately (for the purpose of the Trans actor-critic equilibrium)
        '''
    
    def train_multitrans_actor(self, batch_size):
        self.trans1.train()
        self.trans2.train()
        self.trans3.train()
        self.trans4.train()
        
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        bc_losses1 = []
        bc_losses2 = []
        bc_losses3 = []
        bc_losses4 = []
        
        ascent_losses1 = []
        ascent_losses3 = []
        
        for chunk in chunks:
            # BEHAVIOR CLONNING
            batch = self.trans_RB.sample(chunk)
            states = batch[0]
            targets = batch[2]
            
            preds1 = self.trans1(states)
            preds2 = self.trans2(states)
            preds3 = self.trans3(states)
            preds4 = self.trans4(states)
            
            loss1 = nn.MSELoss()(preds1, targets)
            loss2 = nn.MSELoss()(preds2, targets)
            loss3 = nn.MSELoss()(preds3, targets)
            loss4 = nn.MSELoss()(preds4, targets)
            
            self.trans1_actor_optimizer.zero_grad()
            self.trans2_actor_optimizer.zero_grad()
            self.trans3_actor_optimizer.zero_grad()
            self.trans4_actor_optimizer.zero_grad()
            
            bc_losses1.append(loss1.item())
            bc_losses2.append(loss2.item())
            bc_losses3.append(loss3.item())
            bc_losses4.append(loss4.item())
            
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            
            self.trans1_actor_optimizer.step()
            self.trans2_actor_optimizer.step()
            self.trans3_actor_optimizer.step()
            self.trans4_actor_optimizer.step()
            
            # GRADIENT ASCENT
            trans_loss1 = -self.critic.Q1(states[:,:,-1,:], self.trans1(states)).mean()
            trans_loss3 = -self.critic.Q1(states[:,:,-1,:], self.trans3(states)).mean()
            
            ascent_losses1.append(trans_loss1.cpu().detach().numpy())
            ascent_losses3.append(trans_loss3.cpu().detach().numpy())
            
            self.trans1_actor_optimizer.zero_grad()
            self.trans3_actor_optimizer.zero_grad()
            
            trans_loss1.backward()
            trans_loss3.backward()
            
            self.trans1_actor_optimizer.step()
            self.trans3_actor_optimizer.step()
                
        self.trans_RB.reset()
        self.experiment.add_scalar('Trans_BC_loss1', np.mean(bc_losses1), self.eval_counter)
        self.experiment.add_scalar('Trans_BC_loss2', np.mean(bc_losses2), self.eval_counter)
        self.experiment.add_scalar('Trans_BC_loss3', np.mean(bc_losses3), self.eval_counter)
        self.experiment.add_scalar('Trans_BC_loss4', np.mean(bc_losses4), self.eval_counter)
        
        self.experiment.add_scalar('Trans_Online_loss1', np.mean(ascent_losses1), self.eval_counter)
        self.experiment.add_scalar('Trans_Online_loss3', np.mean(ascent_losses3), self.eval_counter)
        
        self.eval_counter += 1	
            
        
        for param, target_param in zip(self.trans1.parameters(), self.trans_target1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.trans2.parameters(), self.trans_target2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.trans3.parameters(), self.trans_target3.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.trans4.parameters(), self.trans_target4.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)            
        
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()


    def train_multitrans_critic(self, batch_size, additional_bellman):
        self.trans1.train()
        self.trans2.train()
        self.trans3.train()
        self.trans4.train()
        
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        
        bc_losses1 = []
        bc_losses2 = []
        bc_losses3 = []
        bc_losses4 = []
        bellman_losses1 = []
        bellman_losses2 = []
        bellman_losses3 = []
        bellman_losses4 = []
        
        for chunk in chunks:
            # BEHAVIOR CLONNING
            batch = self.trans_RB.sample(chunk)
            states = batch[0]       # n_e, b_s, cont, s_d
            next_states = batch[1]
            actions = batch[2]      # n_e, b_s, a_d
            rewards = batch[3]      # n_e, b_s, 1
            not_dones = batch[4]    # n_e, b_s, 1
            with torch.no_grad():   
                target1, target2 = self.critic(states[:,:,-1,:], actions)
            
            pred1_1, pred2_1 = self.trans1.critic_forward(states, actions)
            pred1_2, pred2_2 = self.trans2.critic_forward(states, actions)
            pred1_3, pred2_3 = self.trans3.critic_forward(states, actions)
            pred1_4, pred2_4 = self.trans4.critic_forward(states, actions)
            
            loss1 = nn.MSELoss()(pred1_1, target1) + nn.MSELoss()(pred2_1, target2)
            loss2 = nn.MSELoss()(pred1_2, target1) + nn.MSELoss()(pred2_2, target2)
            loss3 = nn.MSELoss()(pred1_3, target1) + nn.MSELoss()(pred2_3, target2)
            loss4 = nn.MSELoss()(pred1_4, target1) + nn.MSELoss()(pred2_4, target2)
            
            self.trans1_critic_optimizer.zero_grad()
            self.trans2_critic_optimizer.zero_grad()
            self.trans3_critic_optimizer.zero_grad()
            self.trans4_critic_optimizer.zero_grad()
            
            bc_losses1.append(loss1.item())
            bc_losses2.append(loss2.item())
            bc_losses3.append(loss3.item())
            bc_losses4.append(loss4.item())
            
            loss1.backward()
            loss2.backward()
            loss3.backward()
            loss4.backward()
            
            self.trans1_critic_optimizer.step()
            self.trans2_critic_optimizer.step()
            self.trans3_critic_optimizer.step()
            self.trans4_critic_optimizer.step()
            
            if additional_bellman:
                with torch.no_grad():
                    noise = (
                        torch.randn_like(actions) * self.policy_noise        #noise = (n_e, b_s, a_d)
                    ).clamp(-self.noise_clip, self.noise_clip)
                    
                    next_actions1 = (
                        self.trans_target1(next_states) + noise               #next_action = (n_e, b_s, a_d)
                    ).clamp(-self.max_action, self.max_action)
                    next_actions2 = (
                        self.trans_target2(next_states) + noise               #next_action = (n_e, b_s, a_d)
                    ).clamp(-self.max_action, self.max_action)
                    next_actions3 = (
                        self.trans_target3(next_states) + noise               #next_action = (n_e, b_s, a_d)
                    ).clamp(-self.max_action, self.max_action)
                    next_actions4 = (
                        self.trans_target4(next_states) + noise               #next_action = (n_e, b_s, a_d)
                    ).clamp(-self.max_action, self.max_action)
                    
                    # Compute the target Q value
                    target_Q1_1, target_Q2_1 = self.trans_target1.critic_forward(next_states, next_actions1)
                    target_Q_1 = torch.min(target_Q1_1, target_Q2_1)                          
                    target_Q_1 = rewards + not_dones * self.discount * target_Q_1 
                    # Compute the target Q value
                    target_Q1_2, target_Q2_2 = self.trans_target2.critic_forward(next_states, next_actions2)
                    target_Q_2 = torch.min(target_Q1_2, target_Q2_2)                          
                    target_Q_2 = rewards + not_dones * self.discount * target_Q_2
                    # Compute the target Q value
                    target_Q1_3, target_Q2_3 = self.trans_target3.critic_forward(next_states, next_actions3)
                    target_Q_3 = torch.min(target_Q1_3, target_Q2_3)                          
                    target_Q_3 = rewards + not_dones * self.discount * target_Q_3
                    # Compute the target Q value
                    target_Q1_4, target_Q2_4 = self.trans_target4.critic_forward(next_states, next_actions4)
                    target_Q_4 = torch.min(target_Q1_4, target_Q2_4)                          
                    target_Q_4 = rewards + not_dones * self.discount * target_Q_4
                    
            
                # Get current Q estimates
                current_Q1_1, current_Q2_1 = self.trans1.critic_forward(states, actions)                     #current_Q1(and Q2) = (n_e, b_s, 1)
                # Get current Q estimates
                current_Q1_2, current_Q2_2 = self.trans2.critic_forward(states, actions)
                # Get current Q estimates
                current_Q1_3, current_Q2_3 = self.trans3.critic_forward(states, actions)
                # Get current Q estimates
                current_Q1_4, current_Q2_4 = self.trans4.critic_forward(states, actions)

                # Compute critic loss
                critic_loss_1 = F.mse_loss(current_Q1_1, target_Q_1) + F.mse_loss(current_Q2_1, target_Q_1)
                critic_loss_2 = F.mse_loss(current_Q1_2, target_Q_2) + F.mse_loss(current_Q2_2, target_Q_2)
                critic_loss_3 = F.mse_loss(current_Q1_3, target_Q_3) + F.mse_loss(current_Q2_3, target_Q_3)
                critic_loss_4 = F.mse_loss(current_Q1_4, target_Q_4) + F.mse_loss(current_Q2_4, target_Q_4)

                # Optimize the critic
                self.trans1_critic_optimizer.zero_grad()
                self.trans2_critic_optimizer.zero_grad()
                self.trans3_critic_optimizer.zero_grad()
                self.trans4_critic_optimizer.zero_grad()
                
                bellman_losses1.append(critic_loss_1.item())
                bellman_losses2.append(critic_loss_2.item())
                bellman_losses3.append(critic_loss_3.item())
                bellman_losses4.append(critic_loss_4.item())
                
                critic_loss_1.backward()
                critic_loss_2.backward()
                critic_loss_3.backward()
                critic_loss_4.backward()
                
                self.trans1_critic_optimizer.step()
                self.trans2_critic_optimizer.step()
                self.trans3_critic_optimizer.step()
                self.trans4_critic_optimizer.step()
                
        self.experiment.add_scalar('Trans_Critic_BC_loss1', np.mean(bc_losses1), self.eval_counter)
        self.experiment.add_scalar('Trans_Critic_BC_loss2', np.mean(bc_losses2), self.eval_counter)
        self.experiment.add_scalar('Trans_Critic_BC_loss3', np.mean(bc_losses3), self.eval_counter)
        self.experiment.add_scalar('Trans_Critic_BC_loss4', np.mean(bc_losses4), self.eval_counter)
        if additional_bellman:
            self.experiment.add_scalar('Trans_Critic_Bellman_loss1', np.mean(bellman_losses1), self.eval_counter) 
            self.experiment.add_scalar('Trans_Critic_Bellman_loss2', np.mean(bellman_losses2), self.eval_counter) 
            self.experiment.add_scalar('Trans_Critic_Bellman_loss3', np.mean(bellman_losses3), self.eval_counter) 
            self.experiment.add_scalar('Trans_Critic_Bellman_loss4', np.mean(bellman_losses4), self.eval_counter) 



############################################################################################################################
class ReplayBuffer(object):
    '''
    Реплей буфер для MLP агента, не имеет контекста
    '''
    def __init__(self, num_envs, state_dim, action_dim, max_size=int(1e6)):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((num_envs, max_size, state_dim))
        self.action = np.zeros((num_envs, max_size, action_dim))
        self.next_state = np.zeros((num_envs, max_size, state_dim))
        self.reward = np.zeros((num_envs, max_size, 1))
        self.not_done = np.zeros((num_envs, max_size, 1))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        


    def add(self, state, action, next_state, reward, done):
        self.state[:,self.ptr,] = state
        self.action[:,self.ptr,] = action
        self.next_state[:,self.ptr,] = next_state
        self.reward[:,self.ptr,] = reward.reshape(-1,1)
        self.not_done[:,self.ptr,] = 1. - done.reshape(-1,1)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[:,ind,]).to(self.device),
            torch.FloatTensor(self.action[:,ind,]).to(self.device),
            torch.FloatTensor(self.next_state[:,ind,]).to(self.device),
            torch.FloatTensor(self.reward[:,ind,]).to(self.device),
            torch.FloatTensor(self.not_done[:,ind,]).to(self.device)
        )


class Trans_RB(object):
    def __init__(self, num_envs, size, context, state_dim, act_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context = context
        self.idx = 0
        self.overfilled = False
        self.num_envs = num_envs

        # Инициализация буферов для хранения данных
        self.observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.next_observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32).to(self.device)                   # n_e, size, a_d
        self.rewards = torch.zeros((num_envs, size, 1), dtype=torch.float32).to(self.device)    
        self.not_dones = torch.zeros((num_envs, size, 1), dtype=torch.float32).to(self.device)
    
    def recieve_traj(self, obs, next_obs, acts, rews, n_dones):
        ''' 
        obs list(...,tensor(n_e, ),....)
        '''

        obs = torch.stack(obs, dim=1)    # n_e, cont, s_d
        next_obs = torch.stack(next_obs, dim=1)    # n_e, cont, s_d
        acts = acts.to(torch.float32)    # n_e, a_d
        rews = rews.to(torch.float32) 
        n_dones = n_dones.to(torch.float32)
        
        obs = mean_padding(obs, self.context)
        next_obs = mean_padding(next_obs, self.context)
        
        self.observations[:,self.idx,] = obs.to(device)
        self.next_observations[:,self.idx,] = next_obs.to(device)
        self.actions[:,self.idx] = acts.to(device)
        self.rewards[:,self.idx,] = rews.to(device)
        self.not_dones[:,self.idx,] = n_dones.to(device)

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, idxs):
        
        batch = (
            self.observations[:,idxs, ],  # n_e, b_s, cont, s_d
            self.next_observations[:,idxs, ],
            self.actions[:,idxs, ],        # n_e, b_s, a_d
            self.rewards[:,idxs, ],         # n_e, b_s, 1
            self.not_dones[:,idxs, ]        # n_e, b_s, 1
        )

        return batch

    def reset(self):
        
        # Переинициализация атрибутов как при первом вызове __init__
        self.idx = 0
        self.overfilled = False
        self.observations = torch.zeros((self.num_envs, self.size, self.context, self.state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.actions = torch.zeros((self.num_envs, self.size, self.act_dim), dtype=torch.float32).to(self.device)
        self.rewards = torch.zeros((self.num_envs, self.size, 1), dtype=torch.float32).to(self.device)    
        self.not_dones = torch.zeros((self.num_envs, self.size, 1), dtype=torch.float32).to(self.device)


class New_Trans_RB():
    def __init__(self, num_envs, size, context, state_dim, act_dim, obs_mode):
        self.size = size
        self.context = context
        self.idx = 0
        self.overfilled = False
        self.obs_mode = obs_mode

        self.observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32)
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32)
        self.returns = torch.zeros((num_envs, size, 1), dtype=torch.float32)
        self.dones = torch.zeros((num_envs, size, 1), dtype=torch.float32)
        self.next_observations = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32)
        
        if obs_mode != 'state':
            channels = 3 if obs_mode == 'rgb' else 4
            self.img_observations = torch.zeros((num_envs, size, context, 128, 128, channels), dtype=torch.float32)
            self.img_next_observations = torch.zeros((num_envs, size, context, 128, 128, channels), dtype=torch.float32)
            
    
    def recieve_traj(self, obs, acts, rets, dones, next_obs, img_obs=None, img_next_obs=None):
        '''
        Принимает:
        obs = [....,Tens(n_e, s_d),....]
        acts = Tens(n_e, a_d)
        rets = Tens(n_e, 1)
        dones = Tens(n_e, 1)
        next_obs = [....,Tens(n_e, s_d),....]
        '''

        obs = torch.stack(obs, dim=1)                           # n_e, <=cont, s_d
        obs = mean_padding(obs, self.context)
        next_obs = torch.stack(next_obs, dim=1)                 # n_e, <=cont, s_d
        next_obs = mean_padding(next_obs, self.context)
        
        acts = acts.to(torch.float32)                           # n_e, a_d
        rets = rets.to(torch.float32)                           # n_e, 1
        dones = dones                                           # n_e, 1
        
        
        
        if self.obs_mode != 'state':
            img_obs = torch.stack(img_obs, dim=1).float() 
            img_next_obs = torch.stack(img_next_obs, dim=1).float() 
            img_obs[:, :, :, :, :3] /= 225.0
            img_next_obs[:, :, :, :, :3] /= 225.0
            self.img_observations[:,self.idx,] = img_obs
            self.img_next_observations[:,self.idx,] = img_next_obs
        self.observations[:,self.idx,] = obs
        self.actions[:,self.idx,] = acts
        self.returns[:,self.idx,] = rets
        self.dones[:,self.idx,] = dones
        self.next_observations[:,self.idx,] = next_obs

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, batch_size):
        if batch_size > self.size:
            raise ValueError("Запрошенный размер выборки больше, чем размер буфера.")
        
        elif (batch_size >= self.idx) and (self.overfilled == False):
            idxs = torch.randperm(self.idx) # тогда просто урезаем размер батча
            #raise ValueError("Запрошенный размер выборки {batch_size} больше, чем index {self.idx}.")
        elif (batch_size >= self.idx) and (self.overfilled == True):
            idxs = torch.randperm(self.size)[:batch_size]
        
        elif (batch_size < self.idx) and (self.overfilled == False):
            idxs = torch.randperm(self.idx)[:batch_size]
        
        elif (batch_size < self.idx) and (self.overfilled == True):
            idxs = torch.randperm(self.size)[:batch_size]
        
        
        

        batch = (
            self.observations[:,idxs, ],
            self.actions[:,idxs, ],
            self.returns[:,idxs, ],
            self.dones[:,idxs, ],
            self.next_observations[:,idxs, ],
        ) if self.obs_mode == 'state' else (
            self.observations[:,idxs, ],
            self.actions[:,idxs, ],
            self.returns[:,idxs, ],
            self.dones[:,idxs, ],
            self.next_observations[:,idxs, ],
            self.img_observations[:,idxs, ],
            self.img_next_observations[:,idxs, ]
        ) 


        return batch
        
        

def split_indices(indices, chunk_size):
    # Если размер индексов меньше, чем размер порции, просто возвращаем их как один блок
    if len(indices) <= chunk_size:
        return [indices]
    
    # Разделяем индексы на порции фиксированного размера
    chunks = torch.split(indices, chunk_size)

    return chunks        