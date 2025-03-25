import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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



class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = F.relu(self.l2_2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, obs_mode, conv_lat_dim):
        super(Critic, self).__init__()
        self.obs_mode = obs_mode
        self.conv_lat_dim = conv_lat_dim
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256) if obs_mode == 'state' else nn.Linear(state_dim + conv_lat_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256) if obs_mode == 'state' else nn.Linear(state_dim + conv_lat_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l5_2 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)
    def forward(self, state, action, img_state=None):

        sa = torch.cat([state, action], -1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l5_2(q2))
        q2 = self.l6(q2)
        return q1, q2
    def Q1(self, state, action, img_state=None):
        sa = torch.cat([state, action], -1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)
        return q1

class LSTM_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, num_heads=2, num_layers=1):
        super(LSTM_Critic, self).__init__()
        
        self.hidden_dim = d_model
        self.act_encoder = nn.Linear(state_dim, self.hidden_dim)
        # # Трансформер-энкодер
        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim
        # )
        # self.transformer_encoder = nn.TransformerEncoder(
        #     self.encoder_layer, num_layers=num_layers
        # )
        #
        # 
        # self.transformer_encoder = CustomTransformerEncoder(d_model, num_heads, 512, 0.05, False, False, False, 'GRU', 'Trans')
        
        self.transformer_encoder = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1, batch_first=True)
        
        # Полносвязные слои для Q1 и Q2
        self.fc1 = nn.Linear(d_model + action_dim, d_model)
        self.fc2 = nn.Linear(d_model, 1)

        self.fc3 = nn.Linear(d_model + action_dim, d_model)
        self.fc4 = nn.Linear(d_model, 1)

    def forward(self, state, action):
        n_e, bs, cont, s_d = state.shape
        state = state.view(-1, cont, s_d)  # плющим для аттеншена n_e*bs, context, s_d
        state = self.act_encoder(state)
        # Применяем трансформер-энкодер
        transformer_out = self.transformer_encoder(state)  # n_e*bs, context, d_m
        #transformer_out = transformer_out[:, -1, :]        # n_e*bs, d_m
        transformer_out = transformer_out[0][:, -1, :]
        transformer_out = transformer_out.view(n_e, bs, self.hidden_dim)  # n_e, bs, d_m
        # Q1
        sa = torch.cat([transformer_out, action], dim=-1)   # n_e, bs, d_m+a_d
        q1 = F.relu(self.fc1(sa))
        q1 = self.fc2(q1)

        # Q2
        q2 = F.relu(self.fc3(sa))
        q2 = self.fc4(q2)

        return q1, q2
    
    
    def Q1(self, state, action):
        n_e, bs, cont, s_d = state.shape
        state = state.view(-1, cont, s_d)  # плющим для аттеншена n_e*bs, context, s_d
        state = self.act_encoder(state)
        # Применяем трансформер-энкодер
        transformer_out = self.transformer_encoder(state)  # n_e*bs, context, d_m
        transformer_out = transformer_out[0][:, -1, :]        # n_e*bs, d_m
        transformer_out = transformer_out.view(n_e, bs, self.hidden_dim)  # n_e, bs, d_m
        # Q1
        sa = torch.cat([transformer_out, action], dim=-1)   # n_e, bs, d_m+a_d
        q1 = F.relu(self.fc1(sa))
        q1 = self.fc2(q1)

        return q1  
    
class Trans_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model=256, num_heads=2, num_layers=1):
        super(Trans_Critic, self).__init__()
        
        self.hidden_dim = d_model
        self.act_encoder = nn.Linear(state_dim, self.hidden_dim)
        self.transformer_encoder = CustomTransformerEncoder(d_model, num_heads, 512, 0.05, False, False, False, 'GRU', 'Trans')
        
        
        # Полносвязные слои для Q1 и Q2
        self.fc1 = nn.Linear(d_model + action_dim, d_model)
        self.fc2 = nn.Linear(d_model, 1)

        self.fc3 = nn.Linear(d_model + action_dim, d_model)
        self.fc4 = nn.Linear(d_model, 1)

    def forward(self, state, action):
        n_e, bs, cont, s_d = state.shape
        state = state.view(-1, cont, s_d)  # плющим для аттеншена n_e*bs, context, s_d
        state = self.act_encoder(state)
        # Применяем трансформер-энкодер
        transformer_out = self.transformer_encoder(state)  # n_e*bs, context, d_m
        transformer_out = transformer_out[:, -1, :]        # n_e*bs, d_m
        transformer_out = transformer_out.view(n_e, bs, self.hidden_dim)  # n_e, bs, d_m
        # Q1
        sa = torch.cat([transformer_out, action], dim=-1)   # n_e, bs, d_m+a_d
        q1 = F.relu(self.fc1(sa))
        q1 = self.fc2(q1)

        # Q2
        q2 = F.relu(self.fc3(sa))
        q2 = self.fc4(q2)

        return q1, q2
    
    
    def Q1(self, state, action):
        n_e, bs, cont, s_d = state.shape
        state = state.view(-1, cont, s_d)  # плющим для аттеншена n_e*bs, context, s_d
        state = self.act_encoder(state)
        # Применяем трансформер-энкодер
        transformer_out = self.transformer_encoder(state)  # n_e*bs, context, d_m
        transformer_out = transformer_out[:, -1, :]        # n_e*bs, d_m
        transformer_out = transformer_out.view(n_e, bs, self.hidden_dim)  # n_e, bs, d_m
        # Q1
        sa = torch.cat([transformer_out, action], dim=-1)   # n_e, bs, d_m+a_d
        q1 = F.relu(self.fc1(sa))
        q1 = self.fc2(q1)

        return q1  
    
    
class TD3(object):
    def __init__(
        self,
        num_envs,
        obs_mode,
        context_length,
        model_config,
        state_dim,
        action_dim,
        max_action,
        discount,
        tau,
        policy_noise,
        noise_clip,
        policy_freq,
        grad_clip,
        preload_weights=None
):

        self.context_length = context_length
        
        if preload_weights==None:
            print('No preload weights!')
            print('***INITIALIZING EMPTY MODEL***')
            # self.actor = Actor(state_dim, action_dim, max_action).to(device)
            # self.actor_target = copy.deepcopy(self.actor).to(device)
            # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

            # self.critic = Critic(state_dim, action_dim, obs_mode, model_config['conv_lat_dim']).to(device)
            # self.critic_target = copy.deepcopy(self.critic).to(device)
            # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            
            
            #self.actor = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode=obs_mode).to(device)
            #self.actor_target = copy.deepcopy(self.actor).to(device)
            #self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            
            #self.critic = Trans_Critic(state_dim=state_dim, action_dim=action_dim, d_model=256, num_heads=2, num_layers=1).to(device)
            #self.critic_target = copy.deepcopy(self.critic).to(device)
            #self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            
            model_config['actor_mode'] = 'LSTM'
            self.actor = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode=obs_mode).to(device)
            self.actor_target = copy.deepcopy(self.actor).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
            
            
            self.critic = LSTM_Critic(state_dim=state_dim, action_dim=action_dim, d_model=256, num_heads=2, num_layers=1).to(device)
            self.critic_target = copy.deepcopy(self.critic).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
            
            model_config['actor_mode'] = 'Trans'
            self.trans_actor = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode=obs_mode).to(device)
            self.trans_actor_target = copy.deepcopy(self.trans_actor).to(device)
            self.trans_actor_optimizer = torch.optim.Adam(self.trans_actor.parameters(), lr=3e-4)
            
            self.trans_critic = Trans_Critic(state_dim=state_dim, action_dim=action_dim, d_model=256, num_heads=2, num_layers=1).to(device)
            self.trans_critic_target = copy.deepcopy(self.trans_critic).to(device)
            self.trans_critic_optimizer = torch.optim.Adam(self.trans_critic.parameters(), lr=3e-4)
            self.trans_RB = Trans_RB(num_envs, 30000, context_length, state_dim, action_dim) 
            
        
        elif preload_weights != None:
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
        



        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.context_length = context_length
        self.obs_mode = obs_mode
        self.total_it = 0
        self.eval_counter = 0
        self.trans_critic_mode = model_config['critic_mode']
        self.grad_clip = grad_clip
        

    def select_action(self, state): # state Tens(1, s_d)
        return self.actor(state).cpu().data.numpy()
    
    def train(self, batch_size):
        '''
        Function for accelerator training on the first stage
        '''
        self.total_it += 1
        train_batch = self.new_trans_RB.sample(batch_size)
        if self.obs_mode == 'state':
            states, actions, rewards, dones, next_states = train_batch
        else:
            states, actions, rewards, dones, next_states, img_states, img_next_states = train_batch 
            img_states, img_next_states = img_states.to(device).requires_grad_(True), img_next_states.to(device).requires_grad_(True)	
        
        states = states.to(device).requires_grad_(True)											#n_e, bs, context, state_dim
        actions = actions.to(device).requires_grad_(True)										#n_e, bs, action_dim
        rewards = rewards.to(device).requires_grad_(True)										#n_e, bs, 1
        dones = dones.to(device)											                    #n_e, bs, 1
        next_states = next_states.to(device).requires_grad_(True)								#n_e, bs, context, state_dim

        self.actor.train()
        self.critic.train()    

        with torch.no_grad():
            noise = (                                                           #n_e, bs, a_d
                torch.randn_like(actions) * self.policy_noise  
            ).clamp(-self.noise_clip, self.noise_clip)
            
            if self.obs_mode == 'state':
                next_action = (
                    self.actor_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.actor_target.actor_forward(next_states, img_next_states)
                noise = ( torch.randn_like(next_action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)  
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)                                      #target_Q = (n_e, bs, 1)
            target_Q = rewards + (1-dones) * self.discount * target_Q       #target_Q = (n_e, bs, 1) + (n_e, bs, 1) * const * (n_e, bs, 1)
        
        current_Q1, current_Q2 = self.critic(states, actions)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.experiment.add_scalar('Critic_loss', critic_loss.item(), self.total_it)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)

        
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
        
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            actor_loss = -self.critic.Q1(states, self.actor.actor_forward(states)).mean()    
            
            self.experiment.add_scalar('Actor_loss', actor_loss, self.total_it)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor.parameters(), self.grad_clip)
            actor_grad_norm = sum(p.grad.norm().item() for p in self.actor.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', actor_grad_norm, self.total_it)
            self.actor_optimizer.step()
            
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)    
    
    
    def stage_2_train(self, batch_size):
        '''
        Function for transformer training on the second stage
        '''
        self.total_it += 1
        train_batch = self.new_trans_RB.sample(batch_size)
        if self.obs_mode == 'state':
            states, actions, rewards, dones, next_states = train_batch
        else:
            states, actions, rewards, dones, next_states, img_states, img_next_states = train_batch 
            img_states, img_next_states = img_states.to(device).requires_grad_(True), img_next_states.to(device).requires_grad_(True)	
        
        states = states.to(device).requires_grad_(True)											#n_e, bs, context, state_dim
        actions = actions.to(device).requires_grad_(True)										#n_e, bs, action_dim
        rewards = rewards.to(device).requires_grad_(True)										#n_e, bs, 1
        dones = dones.to(device)											                    #n_e, bs, 1
        next_states = next_states.to(device).requires_grad_(True)								#n_e, bs, context, state_dim

        self.trans_actor.train()
        self.trans_critic.train()    

        with torch.no_grad():
            noise = (                                                           #n_e, bs, a_d
                torch.randn_like(actions) * self.policy_noise  
            ).clamp(-self.noise_clip, self.noise_clip)
            
            if self.obs_mode == 'state':
                next_action = (
                    self.trans_actor_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.trans_actor_target.actor_forward(next_states, img_next_states)
                noise = ( torch.randn_like(next_action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)  
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            target_Q1, target_Q2 = self.trans_critic_target(next_states, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)                                      #target_Q = (n_e, bs, 1)
            target_Q = rewards + (1-dones) * self.discount * target_Q       #target_Q = (n_e, bs, 1) + (n_e, bs, 1) * const * (n_e, bs, 1)
        
        current_Q1, current_Q2 = self.trans_critic(states, actions)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.experiment.add_scalar('Critic_loss', critic_loss.item(), self.total_it)

        self.trans_critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_value_(self.trans_critic.parameters(), self.grad_clip)
        critic_grad_norm = sum(p.grad.norm().item() for p in self.trans_critic.parameters() if p.grad is not None)

        
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
        
        self.trans_critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            trans_loss = -self.trans_critic.Q1(states, self.trans_actor.actor_forward(states)).mean()    
            
            self.experiment.add_scalar('Actor_loss', trans_loss, self.total_it)
            
            # Optimize the actor 
            self.trans_actor_optimizer.zero_grad()
            trans_loss.backward()
            torch.nn.utils.clip_grad_value_(self.trans_actor.parameters(), self.grad_clip)
            trans_grad_norm = sum(p.grad.norm().item() for p in self.trans_actor.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', trans_grad_norm, self.total_it)
            self.trans_actor_optimizer.step()
            
            for param, target_param in zip(self.trans_critic.parameters(), self.trans_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)        

            for param, target_param in zip(self.trans_actor.parameters(), self.trans_actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


    def train_trans_actor(self, batch_size, additional_ascent, additional_bellman):
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
            
            preds = self.trans_actor.actor_forward(states)
            bc_loss = nn.MSELoss()(preds, targets)
            self.trans_actor_optimizer.zero_grad()
            bc_losses.append(bc_loss.item())
            bc_loss.backward()
            self.trans_actor_optimizer.step()
            
            if hasattr(self, 'trans_critic'):  #it means that we train trans_critic by BC (and TD-error)
                with torch.no_grad():
                    tgt_q1, tgt_q2 = self.critic(states, targets) #ne, bs, 1
                pred_q1, pred_q2 = self.trans_critic(states, targets)               #ne, bs, 1
                critic_bc_loss = nn.MSELoss()(pred_q1, tgt_q1) + nn.MSELoss()(pred_q2, tgt_q2)
                critic_bc_losses.append(critic_bc_loss.cpu().detach().numpy())
                self.trans_critic_optimizer.zero_grad()
                critic_bc_loss.backward()
                self.trans_critic_optimizer.step()
                
                if additional_bellman is not None:
                    with torch.no_grad():
                        noise = (   torch.randn_like(targets) * self.policy_noise  
                                ).clamp(-self.noise_clip, self.noise_clip)
                        
                        if additional_bellman == 'Trans': # means that target value is generated by Trans 
                            next_action = (self.trans_actor_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                                    ).clamp(-self.max_action, self.max_action)
                            target_Q1, target_Q2 = self.trans_critic_target(next_states, next_action)
                        
                        elif additional_bellman == 'MLP': # means that target value is generated by MLP
                            next_action = (self.actor(next_states) + noise  			                 #next_action = (n_e, bs, a_d)
                                    ).clamp(-self.max_action, self.max_action)
                            target_Q1, target_Q2 = self.critic_target(next_states, next_action)
                        
                        target_Q = torch.min(target_Q1, target_Q2) 
                        target_Q = rewards + not_dones * self.discount * target_Q  #(n_e, b_s, 1)+(n_e, b_s, 1)*const*(n_e, b_s, 1)=(n_e, b_s, 1)
                        
                    current_Q1, current_Q2 = self.trans_critic(states, targets)
                    bellman_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
                    critic_bellman_losses.append(bellman_loss.cpu().detach().numpy())
                    self.trans_critic_optimizer.zero_grad()
                    bellman_loss.backward()
                    self.trans_critic_optimizer.step()
                    
                    
                    '''
                    TO DO: next_states(ne,bs,cont,sd), reward, not_done ||| or I've already done it before ? 
                    '''
                
                
            if additional_ascent is not None:
                if additional_ascent == 'Trans':      
                    ascent_loss = -self.trans_critic.Q1( states, self.trans_actor.actor_forward(states) ).mean()
                elif additional_ascent == 'MLP':
                    ascent_loss = -self.critic.Q1( states, self.trans_actor.actor_forward(states) ).mean()
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
    


############################################################################################################################

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