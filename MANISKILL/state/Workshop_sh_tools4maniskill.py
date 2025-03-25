import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.flatten import FlattenRGBDObservationWrapper

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from RATE_GTrXL.mem_transformer_v2_GTrXL import Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class StateDictWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(StateDictWrapper, self).__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)  # Получаем obs и info от reset()
        # Преобразуем только obs в словарь с ключом 'state'
        return {'state': obs}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)  # Получаем obs, reward, done, info и т.д.
        # Преобразуем только obs в словарь с ключом 'state'
        return {'state': obs}, reward, terminated, truncated, info 
    

def env_constructor(env_name, num_envs, obs_mode, reconf_freq=None):
    if obs_mode == 'state':
        env_kwargs = dict(obs_mode=obs_mode, sim_backend="gpu", control_mode="pd_joint_delta_pos")
        env = gym.make(env_name, num_envs=num_envs, reconfiguration_freq=reconf_freq, **env_kwargs)
        env = ManiSkillVectorEnv(env, num_envs, ignore_terminations=True, record_metrics=True)
        env = StateDictWrapper(env)
        s_d = env.observation_space.shape[-1]
        a_d = env.action_space.shape[-1]
        return env, s_d, a_d
    elif obs_mode == 'rgb':
        env_kwargs = dict(obs_mode=obs_mode, sim_backend="gpu", control_mode="pd_joint_delta_pos")
        env = gym.make(env_name, num_envs=num_envs, reconfiguration_freq=reconf_freq, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=True)
        env = ManiSkillVectorEnv(env, num_envs=num_envs, ignore_terminations=True, record_metrics=True)
        s_d = env.observation_space['state'].shape[-1]
        a_d = env.action_space.shape[-1]
        return env, s_d, a_d
    elif obs_mode == 'rgbd':
        env_kwargs = dict(obs_mode=obs_mode, sim_backend="gpu", control_mode="pd_joint_delta_pos")
        env = gym.make(env_name, num_envs=num_envs, reconfiguration_freq=reconf_freq, **env_kwargs)
        env = FlattenRGBDObservationWrapper(env, rgb=True, depth=True, state=True)
        env = ManiSkillVectorEnv(env, num_envs=num_envs, ignore_terminations=True, record_metrics=True)
        s_d = env.observation_space['state'].shape[-1]
        a_d = env.action_space.shape[-1]
        return env, s_d, a_d

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
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l2_2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l5_2 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        
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

class Student_Critic(nn.Module):
    def __init__(self, state_dim, action_dim, d_model):
        super(Student_Critic, self).__init__()
        # Q1 architecture
        self.conv_lat_dim = d_model-25 #-22
        
        self.l1 = nn.Linear(self.conv_lat_dim + state_dim + action_dim, 700)
        self.l2 = nn.Linear(700, 700)
        self.l2_2 = nn.Linear(700, 512)
        self.l3 = nn.Linear(512, 1)

        # Q2 architecture
        self.l4 = nn.Linear(self.conv_lat_dim + state_dim + action_dim, 700)
        self.l5 = nn.Linear(700, 700)
        self.l5_2 = nn.Linear(700, 512)
        self.l6 = nn.Linear(512, 1)


    def forward(self, img_vec, state, action):
        
        isa = torch.cat([img_vec, state, action], -1)

        q1 = F.relu(self.l1(isa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(isa))
        q2 = F.relu(self.l5(q2))
        q2 = F.relu(self.l5_2(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, img_vec, state, action):
        
        isa = torch.cat([img_vec, state, action], -1)

        q1 = F.relu(self.l1(isa))
        q1 = F.relu(self.l2(q1))
        q1 = F.relu(self.l2_2(q1))
        q1 = self.l3(q1)
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
        preload_weights=None
):
        if preload_weights == None:
            self.actor = Actor(state_dim, action_dim, max_action).to(device)
            self.actor_target = copy.deepcopy(self.actor).to(device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

            self.critic = Critic(state_dim, action_dim).to(device)
            self.critic_target = copy.deepcopy(self.critic).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

            self.trans = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode='state').to(device)
            self.trans_target = copy.deepcopy(self.trans).to(device)
            self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)
            self.trans_RB = Trans_RB(num_envs, 500, context_length, state_dim, action_dim)######################################
        
        else:
            p2tr = preload_weights[0]
            self.trans = torch.load(p2tr).to(device)
            p2tr_tgt = preload_weights[1]
            self.trans_target = torch.load(p2tr_tgt).to(device)
            self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)
            
            p2cr = preload_weights[2]
            self.critic = torch.load(p2cr).to(device)
            p2cr_tgt = preload_weights[3]
            self.critic_target = torch.load(p2cr_tgt).to(device)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        
        
        
#===================================| IMG VERSION |===================================

        # self.trans = Model(**model_config, state_dim=state_dim, act_dim=action_dim, obs_mode='rgb').to(device)
        # self.trans_target = copy.deepcopy(self.trans).to(device)
        # self.trans_optimizer = torch.optim.Adam(self.trans.parameters(), lr=3e-4)
        # self.trans_RB = Img_Trans_RB(num_envs, 50, context_length, state_dim-7, action_dim)
        
        # self.critic = Student_Critic(state_dim-10, action_dim, model_config['d_model']).to(device)
        # self.critic_target = copy.deepcopy(self.critic).to(device)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

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
        
        

    def select_action(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        
        return self.actor(state).cpu().data.numpy()
    
    def Vec_stage_2_train(self, batch_size):
        '''
        Vector-based transformer training
        '''
        self.total_it += 1
        train_batch = self.new_trans_RB.sample(batch_size)
        
        states, actions, rewards, dones, next_states = train_batch
        
        states = states.to(device).requires_grad_(True)											#n_e, bs, context, state_dim
        actions = actions.to(device).requires_grad_(True)										#n_e, bs, action_dim
        rewards = rewards.to(device).requires_grad_(True)										#n_e, bs, 1
        dones = dones.to(device)											                    #n_e, bs, 1
        next_states = next_states.to(device).requires_grad_(True)								#n_e, bs, context, state_dim

        self.trans.train()
        if hasattr(self, 'critic'):
            self.critic.train()

        with torch.no_grad():
            noise = (                                                           #n_e, bs, a_d
                torch.randn_like(actions) * self.policy_noise  
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                    self.trans_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                ).clamp(-self.max_action, self.max_action)
            
            if hasattr(self, 'critic'):
                target_Q1, target_Q2 = self.critic_target(next_states[:,:,-1,:], next_action)
            else:
                target_Q1, target_Q2 = self.trans_target.critic_forward(next_states, next_action)
            
            target_Q = torch.min(target_Q1, target_Q2)                                      #target_Q = (n_e, bs, 1)
            target_Q = rewards + (1-dones) * self.discount * target_Q       #target_Q = (n_e, bs, 1) + (n_e, bs, 1) * const * (n_e, bs, 1)
        
        if hasattr(self, 'critic'):
            current_Q1, current_Q2 = self.critic(states[:,:,-1,:], actions)  #current_Q1 = (n_e, bs, 1)
        else:
            current_Q1, current_Q2 = self.trans.critic_forward(states, actions)


        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.experiment.add_scalar('Critic_loss', critic_loss.item(), self.total_it)

        if hasattr(self, 'critic'):
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic.parameters(), self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        else:
            self.trans_critic_optimizer.zero_grad()
            critic_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.critic_params, self.grad_clip)
            critic_grad_norm = sum(p.grad.norm().item() for p in self.tr_critic_params if p.grad is not None)

        
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
        
        if hasattr(self, 'critic'):
            self.critic_optimizer.step()
        else:
            self.trans_critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            if hasattr(self, 'critic'):
                trans_loss = -self.critic.Q1(states[:,:,-1,:], self.trans.actor_forward(states)).mean()
            else:
                trans_loss = -self.trans.Q1(states, self.trans.actor_forward(states)).mean()    
            
            self.experiment.add_scalar('Actor_loss', trans_loss, self.total_it)
            
            # Optimize the actor 
            self.trans_optimizer.zero_grad()
            trans_loss.backward()
            #torch.nn.utils.clip_grad_value_(self.actor_params, self.grad_clip)
            trans_grad_norm = sum(p.grad.norm().item() for p in self.trans.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', trans_grad_norm, self.total_it)

            self.trans_optimizer.step()
            
            
            # Update the frozen target models
            if hasattr(self, 'critic'):
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    
    def Img_stage_2_train(self, batch_size):
        '''
        IMG-based transformer training
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

        self.trans.train()
        self.critic.train()

        with torch.no_grad():
            noise = (                                                           #n_e, bs, a_d
                torch.randn_like(actions) * self.policy_noise  
            ).clamp(-self.noise_clip, self.noise_clip)
            
            if self.obs_mode == 'state':
                next_action = (
                    self.trans_target.actor_forward(next_states) + noise  			 #next_action = (n_e, bs, a_d)
                ).clamp(-self.max_action, self.max_action)
            else:
                next_action = self.trans_target.actor_forward(next_states, img_next_states)
                noise = ( torch.randn_like(next_action) * self.policy_noise ).clamp(-self.noise_clip, self.noise_clip)  
                next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
                
            n_e, bs, cont, h, w, c = img_next_states.shape
            next_conv_vec = self.trans.convolution(img_next_states.reshape(n_e*bs*cont, c, h, w)).reshape(n_e, bs, cont, self.critic.conv_lat_dim).to(device)
            conv_vec = self.trans.convolution(img_states.reshape(n_e*bs*cont, c, h, w)).reshape(n_e, bs, cont, self.critic.conv_lat_dim).to(device)
            
            
            target_Q1, target_Q2 = self.critic_target( next_conv_vec[:,:,-1,], next_states[:,:,-1,:], next_action )
            target_Q = torch.min(target_Q1, target_Q2)                                      #target_Q = (n_e, bs, 1)
            target_Q = rewards + (1-dones) * self.discount * target_Q       #target_Q = (n_e, bs, 1) + (n_e, bs, 1) * const * (n_e, bs, 1)

        current_Q1, current_Q2 = self.critic(conv_vec[:,:,-1,], states[:,:,-1,:], actions)  #current_Q1 = (n_e, bs, 1)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.experiment.add_scalar('Critic_loss', critic_loss.item(), self.total_it)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        critic_grad_norm = sum(p.grad.norm().item() for p in self.critic.parameters() if p.grad is not None)
        self.experiment.add_scalar('critic_grad_norm', critic_grad_norm, self.total_it)
    
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            if self.obs_mode == 'state':
                trans_loss = -self.critic.Q1(states[:,:,-1,:], self.trans.actor_forward(states)).mean()
            else:
                trans_loss = -self.critic.Q1(conv_vec[:,:,-1,], states[:,:,-1,:], self.trans.actor_forward(states, img_states)).mean()  
            self.experiment.add_scalar('Actor_loss', trans_loss, self.total_it)
            
            # Optimize the actor 
            self.trans_optimizer.zero_grad()
            trans_loss.backward()
            trans_grad_norm = sum(p.grad.norm().item() for p in self.trans.parameters() if p.grad is not None)
            self.experiment.add_scalar('actor_grad_norm', trans_grad_norm, self.total_it)

            self.trans_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)






    def train(self, replay_buffer, batch_size=256):
        self.total_it += 1

        # Sample replay buffer 
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
        

        with torch.no_grad():
            
            noise = (
                torch.randn_like(action) * self.policy_noise        #noise = (n_e, b_s, a_d)
            ).clamp(-self.noise_clip, self.noise_clip)
            
            next_action = (
                self.actor_target(next_state) + noise               #next_action = (n_e, b_s, a_d)
            ).clamp(-self.max_action, self.max_action)
            
            
            

            # Compute the target Q value
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)                          #target_Q = (n_e, b_s, 1)
            target_Q = reward + not_done * self.discount * target_Q             #target_Q = (n_e, b_s, 1) + (n_e, b_s, 1) * const * (n_e, b_s, 1)

        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(state, action)                     #current_Q1(and Q2) = (n_e, b_s, 1)

        # Compute critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q) #успешно сплюснул в константу

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:

            # Compute actor losse
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()    #state=(n_e,b_s,s_d), actor(state)=(n_e,b_s,a_d), actor_loss=(n_e,b_s,1)
            
            # Optimize the actor 
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def train_trans(self, policy, batch_size, experiment, additional_ascent=False):
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        losses = []
        ascent_losses = []
        for chunk in chunks:
            batch = policy.trans_RB.sample(chunk)
            states = batch[0].to(device)        # n_e, b_s, cont, s_d
            actions = batch[1].to(device)           # n_e, b_s, a_d
            
            
            preds = self.trans.actor_forward(states)   # n_e, b_s, s_d
            loss = nn.MSELoss()(preds, actions)
            policy.trans_optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
            policy.trans_optimizer.step()
            
            # GRADIENT ASCENT
            if additional_ascent:      #target_Q = torch.min(target_Q1, target_Q2) 
                trans_loss = -self.critic.Q1(states[:,:,-1,:], self.trans.actor_forward(states)).mean()
                ascent_losses.append(trans_loss.cpu().detach().numpy())
                self.trans_optimizer.zero_grad()
                trans_loss.backward()
                self.trans_optimizer.step()
                
        policy.trans_RB.reset()
        experiment.add_scalar('Trans_loss', np.mean(losses), self.eval_counter)
        self.eval_counter += 1

        for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
    def img_train_trans(self, policy, batch_size, experiment):
        idxs = torch.randperm(self.trans_RB.idx)
        chunks = split_indices(idxs, batch_size)
        losses = []
        crit_losses = []
        for chunk in chunks:
            batch = policy.trans_RB.sample(chunk)
            img_states = batch[0].to(device)        # n_e, b_s, cont, 128, 128, 3
            vec_states = batch[1].to(device)        # n_e, b_s, cont, s_d-10
            actions = batch[2].to(device)           # n_e, b_s, a_d
            full_states = batch[3].to(device)       # n_e, b_s, s_d
            
            preds = self.trans.actor_forward(vec_states, img_states)
            loss = nn.MSELoss()(preds, actions)
            policy.trans_optimizer.zero_grad()
            losses.append(loss.item())
            loss.backward()
            policy.trans_optimizer.step()
            
            tgt_q1, tgt_q2 = self.critic(full_states, actions)  # n_e, b_s, 1
            with torch.no_grad():
                n_e, bs, cont, h, w, c = img_states.shape
                conv_vec = self.trans.convolution(img_states.reshape(n_e*bs*cont, c, h, w)).reshape(n_e, bs, cont, self.student_critic.conv_lat_dim) # -3 # n_e, b_s, conv.lat.dim
            pred_q1, pred_q2 = self.student_critic(conv_vec[:,:,-1,], vec_states[:,:,-1,], actions)
            crit_loss = nn.MSELoss()(pred_q1, tgt_q1) + nn.MSELoss()(pred_q2, tgt_q2)
            self.student_critic_optimizer.zero_grad()
            crit_losses.append(crit_loss.item())
            crit_loss.backward()
            self.student_critic_optimizer.step()
            
            
        policy.trans_RB.reset()
        experiment.add_scalar('Trans_loss', np.mean(losses), self.eval_counter)
        experiment.add_scalar('St_Critic_loss', np.mean(crit_losses), self.eval_counter)
        self.eval_counter += 1

        for param, target_param in zip(self.trans.parameters(), self.trans_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.student_critic.parameters(), self.student_critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)    

        

############################################################################################################################
class ReplayBuffer(object):
    
    def __init__(self, num_envs, state_dim, action_dim, max_size=int(5e5)):
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
    '''
    State-based transformer replay buffer
    '''
    def __init__(self, num_envs, size, context, state_dim, act_dim):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.size = size
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.context = context
        self.idx = 0
        self.overfilled = False
        self.num_envs = num_envs

        
        self.states = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32).to(self.device)                   # n_e, size, a_d
        
        
    
    def recieve_traj(self, sts, acts):
        ''' 
        obs list(...,tensor(n_e, 1, 128, 128, 3),....)
        '''

        sts = torch.stack(sts, dim=1)    # n_e, cont, s_d
        acts = acts.to(torch.float32)    # n_e, a_d
        
        self.states[:,self.idx,] = sts.to(device)
        self.actions[:,self.idx] = acts.to(device)

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, idxs):
        
        batch = (
            self.states[:,idxs, ],          # n_e, b_s, cont, s_d
            self.actions[:,idxs, ]        # n_e, b_s, a_d
        )

        return batch

    def reset(self):
        
        # Переинициализация атрибутов как при первом вызове __init__
        self.idx = 0
        self.overfilled = False
        self.states = torch.zeros((self.num_envs, self.size, self.context, self.state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.num_envs, self.size, self.act_dim), dtype=torch.float32).to(self.device)                       # n_e, size, a_d


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

        obs = torch.stack(obs, dim=1)                           # n_e, cont, s_d
        acts = acts.to(torch.float32)                           # n_e, a_d
        rets = rets.to(torch.float32)                           # n_e, 1
        dones = dones                                           # n_e, 1
        next_obs = torch.stack(next_obs, dim=1)                 # n_e, cont, s_d
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
        
        
class Img_Trans_RB(object):
    '''
    Image-based transformer replay buffer
    '''
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
        self.observations = torch.zeros((num_envs, size, context, 128, 128, 3), dtype=torch.float32).to(self.device)  # n_e, size, cont, 128, 128, 3
        self.states = torch.zeros((num_envs, size, context, state_dim), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.actions = torch.zeros((num_envs, size, act_dim), dtype=torch.float32).to(self.device)                   # n_e, size, a_d
        
        self.full_states = torch.zeros((num_envs, size, state_dim+10), dtype=torch.float32).to(self.device) #+7  # n_e, size, cont, s_d
        
    
    def recieve_traj(self, obs, sts, acts, full_sts):
        ''' 
        obs list(...,tensor(n_e, 1, 128, 128, 3),....)
        '''

        obs = torch.stack(obs, dim=1)    # n_e, cont, 128, 128, 3
        sts = torch.stack(sts, dim=1)    # n_e, cont, s_d
        acts = acts.to(torch.float32)    # n_e, a_d
        full_sts = full_sts.to(torch.float32)    # n_e, s_d+10
        
        self.observations[:,self.idx,] = obs.to(device)
        self.states[:,self.idx,] = sts.to(device)
        self.actions[:,self.idx] = acts.to(device)
        self.full_states[:,self.idx,] = full_sts.to(device)

        self.idx += 1
        if self.idx >= self.size:
            self.idx = 0
            self.overfilled = True

    def sample(self, idxs):
        
        batch = (
            self.observations[:,idxs, ]/255.0,  # n_e, b_s, cont, 128, 128, 3
            self.states[:,idxs, ],          # n_e, b_s, cont, s_d
            self.actions[:,idxs, ],        # n_e, b_s, a_d
            self.full_states[:,idxs, ]      # n_e, b_s, s_d+10
        )

        return batch

    def reset(self):
        
        # Переинициализация атрибутов как при первом вызове __init__
        self.idx = 0
        self.overfilled = False
        self.observations = torch.zeros((self.num_envs, self.size, self.context, 128, 128, 3), dtype=torch.float32).to(self.device)  # n_e, size, cont, s_d
        self.states = torch.zeros((self.num_envs, self.size, self.context, self.state_dim), dtype=torch.float32).to(self.device)
        self.actions = torch.zeros((self.num_envs, self.size, self.act_dim), dtype=torch.float32).to(self.device)                       # n_e, size, a_d
        self.full_states = torch.zeros((self.num_envs, self.size, self.state_dim+10), dtype=torch.float32).to(self.device) #+7




def split_indices(indices, chunk_size):
    # Если размер индексов меньше, чем размер порции, просто возвращаем их как один блок
    if len(indices) <= chunk_size:
        return [indices]
    
    # Разделяем индексы на порции фиксированного размера
    chunks = torch.split(indices, chunk_size)

    return chunks