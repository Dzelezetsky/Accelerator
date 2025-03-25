import numpy as np
import torch
#import gym
import argparse
import os
import yaml
from Workshop_sh_tools4maniskill import TD3, New_Trans_RB, env_constructor
import copy
#########################################################
from collections import defaultdict

import mani_skill.envs
import gymnasium as gym
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
########################################################

from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


evaluations = []

def eval_transformer(policy, args, eval_episodes, context):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=1)
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    policy.trans.eval()

    for _ in range(eval_episodes):
        
        state, info = eval_env.reset(seed=args.seed) #state (n_e, s_d)
        state = state['state']
        #states = torch.from_numpy(state[0]).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        states = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
        
        t = -1
        while "final_info" not in info:
            t+=1
            if states.shape[1] > context :                      # policy.context_length
                states = states[:, -context:, :]  #   policy.context_length

            s = states.unsqueeze(1)             #s (n_e, 1, cont, s_d)
            sampled_action = policy.trans.actor_forward(s, show_percentage=True)  # sampled_action (n_e, 1, a_d)
            
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
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
    print("---------------------------------------")
    
    policy.trans.train()
    
    return avg_reward, avg_sr_end, avg_sr_once


def second_stage(policy, config, args, experiment):
    
    env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=None)
    env.action_space.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    context_length = config['train_config']['context_length']

    policy.new_trans_RB = New_Trans_RB(args.num_envs, config['train_config']['replay_buffer_size'], context_length, state_dim, action_dim, 'state')
    policy.experiment = experiment
    policy.trans.train()
    policy.critic.train()

    out_states = []
    out_actions = []
    out_rewards = []
    out_dones = []
    
    state, _ = env.reset(seed=args.seed)
    state = state['state']
    truncated = False
    out_states.append( state )
    states = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)

    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    
    episode_num = 0
    eval_counter = 0
    
    avg_reward, avg_sr_end, avg_sr_once = eval_transformer(policy, args, 1, context_length)  #6
    experiment.add_scalar('Eval_reward', avg_reward, 0)
    experiment.add_scalar('Eval_sr_end', avg_sr_end, 0)
    experiment.add_scalar('Eval_sr_once', avg_sr_once, 0)

    for t in range(int(args.max_timesteps)):

        if states.shape[1] > context_length : 
            states = states[:, -context_length:, :]

        s = states.unsqueeze(1)    # n_e, 1, cont, s_d
        if t < args.start_timesteps:
            action = env.action_space.sample()  #sampled_action=arr(n_e, a_d)
            out_actions.append(torch.Tensor(action))
        else:
            sampled_action = policy.trans.actor_forward(s)                  #sampled_action=Tensor(n_e, 1, a_d)
            sampled_action = sampled_action.detach().cpu()[:,0,]            #sampled_action=Tensor(n_e, a_d)
            out_actions.append(sampled_action)
            action = np.clip( sampled_action.numpy(), -1, 1)                #action=Arr(n_e, a_d) 

        state, reward, terminated, truncated, info = env.step( action )
        state = state['state']

        # Store data in replay buffer
        out_dones.append(truncated.to(float).reshape(-1, 1))    #truncated.reshape(-1, 1) = (n_e, 1)
        out_rewards.append(reward.reshape(-1, 1))               #reward.reshape(-1, 1) = (n_e, 1)
        out_states.append( state )

        if t >= context_length-1 and states.shape[1] == context_length:
            states2RB = out_states[-context_length-1:-1]
            act2RB = out_actions[-1]
            ret2RB = out_rewards[-1]
            done2RB = out_dones[-1]
            next_states2RB = out_states[-context_length:]
            policy.new_trans_RB.recieve_traj(states2RB, act2RB, ret2RB, done2RB, next_states2RB)

        cur_state = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  # (n_e, 1, s_d)
        states = torch.cat([states, cur_state], dim=1)
        
        
        if t >= args.start_timesteps:
            policy.Vec_stage_2_train(args.batch_size)
               

        if "final_info" in info:
            avg_sr_once = info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
            avg_sr_end = info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
            avg_ret = info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs
            print(f"Total T: {t+1} Episode Num: {episode_num+1} SR_END : {avg_sr_end:.2f} SR_ONCE : {avg_sr_once:.2f} Reward: {avg_ret:.3f}")
            
            
            out_states = []
            out_actions = []
            out_rewards = []
            out_dones = []
            truncated = False
            
            state, _ = env.reset(seed=args.seed)
            state = state['state']
            out_states.append( state )
            states = copy.deepcopy(state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)

            episode_num += 1


        # Evaluate episode
        if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
            eval_counter += 1
            avg_reward, avg_sr_end, avg_sr_once = eval_transformer(policy, args, 1, context_length)  #6
            experiment.add_scalar('Eval_reward', avg_reward, t)
            experiment.add_scalar('Eval_sr_end', avg_sr_end, t)
            experiment.add_scalar('Eval_sr_once', avg_sr_once, t)
            
            
            
    torch.save(policy.trans, f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST2]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")


if __name__ == "__main__":

    with open("Shuttle/sh_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="PushCube-v1")          # OpenAI gym environment name
    parser.add_argument("--num_envs", default=50, type=int)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--use_train_data", default=True, type=bool)       #|False|  |True|  |False|  |True|
    parser.add_argument("--additional_ascent", default=False, type=bool)    #|False|  |False| |True|   |True|
    parser.add_argument("--start_timesteps", default=0, type=int)# 2e3
    parser.add_argument("--eval_freq", default=3e2, type=int)       # 2e2
    parser.add_argument("--max_timesteps", default=60e3, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)      # 256
    parser.add_argument("--discount", default=0.8, type=float)     # 0.99
    parser.add_argument("--tau", default=0.01, type=float)         # 0.005
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    
    
    for seed in [2,3]:
        args.seed = seed
        path2run = f"WORKSHOP_RUNS/{args.env}/[NEW_ST_ST2]|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|"
    
        experiment = SummaryWriter(log_dir=path2run)
    
        test_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'state', reconf_freq=1)
    
        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": 1,
            "discount": args.discount,
            "tau": args.tau,
        }
    
        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * 1
            kwargs["noise_clip"] = args.noise_clip * 1
            kwargs["policy_freq"] = args.policy_freq
            
        
    
        ######### ↓↓↓↓↓ DOWNLOADING WEIGHTS ↓↓↓↓↓ ######### 
    
        pth2trans = f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2trans_tgt = f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]Trans(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2critic = f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]St_Critic|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        pth2critic_tgt = f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]St_Critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth"
        
        kwargs["preload_weights"] = [pth2trans, pth2trans_tgt, pth2critic, pth2critic_tgt]
        policy = TD3(args.num_envs, 'state', config['train_config']['context_length'], config['model_config'], **kwargs)
        
        second_stage(policy, config, args, experiment)    