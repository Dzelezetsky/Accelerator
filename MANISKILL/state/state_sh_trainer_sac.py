import numpy as np
import torch
#import gym
import argparse
import os
import yaml
from state_sh_tools_sac import SAC, ReplayBuffer, env_constructor

import copy

##########################################
from collections import defaultdict

import mani_skill.envs
import gymnasium as gym
#from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from collections import defaultdict
##########################################
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("MANISKILL/state/sh_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

LOG_STD_MIN = -5
LOG_STD_MAX = 2

def eval_transformer(policy, args, eval_episodes=1):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=1)
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    policy.trans.eval()

    for ep in range(eval_episodes):
        
        state, info = eval_env.reset(seed=args.seed) #state (n_e, s_d)
        state = state['state']
        #states = torch.from_numpy(state[0]).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
        
        t = -1
        while "final_info" not in info:
            t+=1
            if states.shape[1] > policy.context_length : 
                states = states[:, -policy.context_length:, :]

            s = states.unsqueeze(1)             #s (n_e, 1, cont, s_d)
            sampled_action = policy.trans_actor(s, deterministic=True, with_logprob=False) # sampled_action (n_e, 1, a_d)
            sampled_action = sampled_action.detach().cpu()
            action = np.clip( sampled_action.numpy()[:,0,], -1, 1)  #action (n_e, a_d)

            state, r, terminated, truncated, info = eval_env.step( action )
            state = state['state']
            
            cur_state = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  #cur_state (n_e, 1, s_d)
            states = torch.cat([states, cur_state], dim=1)                                          #states (n_e, cont+1, s_d)

        avg_sr_once += info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
        avg_sr_end += info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
        avg_reward += info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs 
            

    avg_reward /= eval_episodes
    avg_sr_end /= eval_episodes
    avg_sr_once /= eval_episodes

    print("---------------------------------------")
    print(f"seed|{args.seed}|Transformer evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
    print("---------------------------------------")
    
    policy.trans.train()
    
    return avg_reward, avg_sr_end, avg_sr_once


def eval_policy(policy, args, eval_episodes=1):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=1)
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    
    vec_out_states = []
    #img_out_states = [] vec_out_states = []
    #vec_out_states = [] out_actions = []
    out_actions = []
    
    
    for ep in range(eval_episodes):
        state, info = eval_env.reset(seed=args.seed)
        state = state['state']
        vec_out_states.append(state)
        t=0
        
        while "final_info" not in info:
            t+=1
            action = policy.select_action(state, deterministic=True)  # action: arr (n_e, a_d)
            
            out_actions.append(torch.Tensor(action))  # Tens (n_e, a_d)
            
            state, r, terminated, truncated, info = eval_env.step(action)
            state = state['state']
            vec_out_states.append(state)
            
            if (t >= policy.context_length):
                vec_states2RB = vec_out_states[-policy.context_length-1:-1]
                act2RB = out_actions[-1]
                policy.trans_RB.recieve_traj(vec_states2RB, act2RB)
            
        avg_sr_once += info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
        avg_sr_end += info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
        avg_reward += info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs 

    avg_reward /= eval_episodes
    avg_sr_end /= eval_episodes
    avg_sr_once /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
    print("---------------------------------------")
    
    
    
    return avg_reward, avg_sr_end, avg_sr_once


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="PushCube-v1")
    parser.add_argument("--num_envs", default=50, type=int)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--use_train_data", default=True, type=bool)
    parser.add_argument("--additional_ascent", default=False, type=bool)  
    parser.add_argument("--start_trans_train", default=0, type=int)
    parser.add_argument("--start_timesteps", default=500, type=int)
    parser.add_argument("--eval_freq", default=600, type=int)
    parser.add_argument("--max_timesteps", default=70000, type=int)
    parser.add_argument("--batch_size", default=600, type=int)
    parser.add_argument("--discount", default=0.8, type=float)
    parser.add_argument("--tau", default=0.01, type=float)
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument("--load_model", default="")
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    
    for RUN in [args.seed]:
        
        path2run = f"ECAI_SAC_CAMERA_READY_ST1/{args.env}/seed={args.seed}"
        experiment = SummaryWriter(log_dir=path2run)
        
        env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=None)
        

        # Set seeds
        #env.seed(args.seed)
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        max_action = float(env.action_space.high[0][0])

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
        }

        policy = SAC(args.num_envs, 'state', config['train_config']['context_length'],
             config['model_config'],
             state_dim, action_dim, max_action,
             discount=args.discount, tau=args.tau)
        
        policy.experiment = experiment
        replay_buffer = ReplayBuffer(args.num_envs, state_dim, action_dim)
        
###################################
        vec_out_states = []
        out_actions = []
###################################      
        
        state, _ = env.reset(seed=args.seed)  #tensor n_e, s_d
        state = state['state']
        if args.use_train_data:
            vec_out_states.append(state)#####################################
        
        truncated = False
        episode_num = 0
        eval_counter = 0
        trans_eval_counter = 0
        max_trans_reward = 0
        for t in range(int(args.max_timesteps)):
            

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()  # Array n_e, a_d
            else:
                action = policy.actor(state, deterministic=False, with_logprob=False).detach().cpu().numpy()      # arr n_e, a_d
                
                

            
            
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = next_state['state']
            done_bool = truncated.to(float)
            
#############################################################
            if args.use_train_data:
                vec_out_states.append(next_state)
                out_actions.append(torch.Tensor(action))################################
                if (t >= policy.context_length):
                    vec_states2RB = vec_out_states[-policy.context_length-1:-1]
                    act2RB = out_actions[-1]
                    policy.trans_RB.recieve_traj(vec_states2RB, act2RB)

################################################################            

            # Store data in replay buffer
            replay_buffer.add(state.cpu(), action, next_state.cpu(), reward.cpu(), done_bool.cpu())

            state = next_state

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
                if args.use_train_data:
                    policy.train_trans_actor(256, additional_ascent=args.additional_ascent)
            if "final_info" in info: 
                avg_sr_once = info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
                avg_sr_end = info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
                avg_ret = info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs
                
                print(f"Total T: {t+1} Episode Num: {episode_num+1} SR_END : {avg_sr_end:.2f} SR_ONCE : {avg_sr_once:.2f} Reward: {avg_ret:.3f}")
                
                state, _ = env.reset(seed=args.seed)
                state = state['state']
                truncated = False
                episode_num += 1 

            # Evaluate episode
            if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
                eval_counter += 1
                avg_reward, avg_sr_end, avg_sr_once = eval_policy(policy, args, 10)
                tr_avg_reward, tr_avg_sr_end, tr_avg_sr_once = eval_transformer(policy, args, 1)
                
                experiment.add_scalar('Eval_reward', avg_reward, t)
                experiment.add_scalar('Eval_sr_end', avg_sr_end, t)
                experiment.add_scalar('Eval_sr_once', avg_sr_once, t)
                
                experiment.add_scalar('Trans_Eval_reward', tr_avg_reward, t)
                experiment.add_scalar('Trans_Eval_sr_end', tr_avg_sr_end, t)
                experiment.add_scalar('Trans_Eval_sr_once', tr_avg_sr_once, t)
                    
                if avg_reward > args.start_trans_train:
                    args.eval_freq = 200
                    policy.train_trans(policy, 256, experiment, additional_ascent=args.additional_ascent) #256
                    
                    
                    if tr_avg_reward > max_trans_reward:
                        max_trans_reward = tr_avg_reward
                        torch.save(policy.trans_actor, f"ECAI_SAC_CAMERA_READY_WEIGHTS[MDP]/{args.env}-[FINAL_ST1]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                        torch.save(policy.trans_actor_target, f"ECAI_SAC_CAMERA_READY_WEIGHTS[MDP]/{args.env}-[FINAL_ST1]Trans(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                        torch.save(policy.critic, f"ECAI_SAC_CAMERA_READY_WEIGHTS[MDP]/{args.env}-[FINAL_ST1]St_Critic|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                        torch.save(policy.critic_target, f"ECAI_SAC_CAMERA_READY_WEIGHTS[MDP]/{args.env}-[FINAL_ST1]St_Critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                else:
                    policy.trans_RB.reset()    
                
                
                
            
              
        
        # torch.save(policy.trans, f"Maniskill_weights/{args.env}/[ST_ST1]Trans|seed={args.seed}|ne={args.num_envs}|.pth")
        # torch.save(policy.trans_target, f"Maniskill_weights/{args.env}/[ST_ST1]Trans(t)|seed={args.seed}|ne={args.num_envs}|.pth")
                        
        # torch.save(policy.critic, f"Maniskill_weights/{args.env}/[ST_ST1]Critic|seed={args.seed}|ne={args.num_envs}|.pth")
        # torch.save(policy.critic_target, f"Maniskill_weights/{args.env}/[ST_ST1]Critic(t)|seed={args.seed}|ne={args.num_envs}|.pth")
        
           