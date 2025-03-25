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


# def img_eval_transformer(policy, args, eval_episodes=1):
#     '''
    
#     Transformers evaluation based on images and vector states.
#     Agent recieves tensors (n_e, 1, context, 128, 128, 3/4) and (n_e, 1, context, s_d)
#     ----- and returns tensor (n_e, 1, a_d)
    
#     '''
#     eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'rgb', reconf_freq=1)
#     avg_reward = 0.
#     avg_sr_end = 0.
#     avg_sr_once= 0.
#     policy.trans.eval()

#     for _ in range(eval_episodes):
#         state, info = eval_env.reset(seed=args.seed) 
        
#         img_state = state['rgb']/255.0            #img_state (n_e, 128, 128, 4/3)
#         vec_state = state['state']            #img_state (n_e, s_d)
#         img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #states (n_e, 1, 128, 128, 4/3)
#         vec_states = copy.deepcopy(vec_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #states (n_e, 1, s_d)
        
#         t = -1
#         while "final_info" not in info:
#             t+=1
#             if vec_states.shape[1] > policy.context_length :
#                 img_states = img_states[:, -policy.context_length:, :]
#                 vec_states = vec_states[:, -policy.context_length:, :]

#             img_s = img_states.unsqueeze(1)              #s (n_e, 1, cont, 128, 128, 4/3)
#             vec_s = vec_states.unsqueeze(1)              #s (n_e, 1, cont, s_d)
#             sampled_action = policy.trans.actor_forward(vec_s, img_s)  # sampled_action Tens(n_e, 1, a_d)
#             sampled_action = sampled_action.detach().cpu()
#             action = np.clip( sampled_action.numpy()[:,0,], -1, 1)  #action Arr(n_e, a_d)

#             state, r, terminated, truncated, info = eval_env.step( action )
            
#             print(action)
            
#             img_state = state['rgb']/255.0            
#             vec_state = state['state']           
#             img_cur_state = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #img_cur_state (n_e, 1, 128, 128, 4/3)
#             vec_cur_state = copy.deepcopy(vec_state).unsqueeze(1).to(device=device, dtype=torch.float32)
#             vec_states = torch.cat([vec_states, vec_cur_state], dim=1)
#             img_states = torch.cat([img_states, img_cur_state], dim=1)                                          #img_states (n_e, cont, 128, 128, 4/3)

#         avg_sr_once += info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
#         avg_sr_end += info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
#         avg_reward += info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs 
            

#     avg_reward /= eval_episodes
#     avg_sr_end /= eval_episodes
#     avg_sr_once /= eval_episodes

#     print("---------------------------------------")
#     print(f"Transformer Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
#     print("---------------------------------------")
    
#     policy.trans.train()
    
#     return avg_reward, avg_sr_end, avg_sr_once

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
    print(f"seed|{args.seed}|Transformer evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
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

    out_st_states = []
    out_img_states = []
    out_actions = []
    out_rewards = []
    out_dones = []
    
    state, _ = env.reset(seed=args.seed)
    st_state = state['state']
    #img_state = state[f"{args.obs_mode}"]/255.0
    truncated = False
    out_st_states.append( st_state )
    #out_img_states.append( img_state )
    st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
    #img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32) #state (n_e, 1, 128, 128, 3)
    
    
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    
    episode_num = 0
    eval_counter = 0

    
    #avg_reward, avg_sr_end, avg_sr_once = img_eval_transformer(policy, args, 1)
    avg_reward, avg_sr_end, avg_sr_once = eval_transformer(policy, args, 1)
    experiment.add_scalar('Eval_reward', avg_reward, eval_counter)
    experiment.add_scalar('Eval_sr_end', avg_sr_end, eval_counter)
    experiment.add_scalar('Eval_sr_once', avg_sr_once, eval_counter)
    
    for t in range(int(args.max_timesteps)):

        if st_states.shape[1] > context_length : 
            st_states = st_states[:, -context_length:, :]
            #img_states = img_states[:, -context_length:, :]

        st_s = st_states.unsqueeze(1)    # n_e, 1, cont, s_d
        #img_s = img_states.unsqueeze(1)    # n_e, 1, cont, h,w,c
        if t < args.start_timesteps:
            action = env.action_space.sample()  #sampled_action=arr(n_e, a_d)
            out_actions.append(torch.Tensor(action))
        else:
            sampled_action = policy.trans.actor_forward(st_s, img_s)                  #sampled_action=Tensor(n_e, 1, a_d)
            sampled_action = sampled_action.detach().cpu()[:,0,]            #sampled_action=Tensor(n_e, a_d)
            out_actions.append(sampled_action)
            action = np.clip( sampled_action.numpy(), -1, 1)                #action=Arr(n_e, a_d) 

        state, reward, terminated, truncated, info = env.step( action )
        st_state = state['state']
        #img_state = state[f"{args.obs_mode}"]/255.0

        # Store data in replay buffer
        out_dones.append(truncated.to(float).reshape(-1, 1))    #truncated.reshape(-1, 1) = (n_e, 1)
        out_rewards.append(reward.reshape(-1, 1))               #reward.reshape(-1, 1) = (n_e, 1)
        out_st_states.append( st_state )
        #out_img_states.append( img_state )

        if t >= context_length-1 and st_states.shape[1] == context_length:
            st_states2RB = out_st_states[-context_length-1:-1]
            #img_states2RB = out_img_states[-context_length-1:-1]
            act2RB = out_actions[-1]
            ret2RB = out_rewards[-1]
            done2RB = out_dones[-1]
            next_st_states2RB = out_st_states[-context_length:]
            #next_img_states2RB = out_img_states[-context_length:]
            #policy.new_trans_RB.recieve_traj(st_states2RB, act2RB, ret2RB, done2RB, next_st_states2RB, img_states2RB, next_img_states2RB)
            policy.new_trans_RB.recieve_traj(st_states2RB, act2RB, ret2RB, done2RB, next_st_states2RB )
            
        st_cur_state = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  # (n_e, 1, s_d)
        #img_cur_state = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  # (n_e, 1, h,w,c)
        st_states = torch.cat([st_states, st_cur_state], dim=1)
        #img_states = torch.cat([img_states, img_cur_state], dim=1)
        
        
        if t >= 1000: #args.start_timesteps
            policy.stage_2_train(args.batch_size)
               

        if "final_info" in info:
            avg_sr_once = info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
            avg_sr_end = info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
            avg_ret = info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs
            print(f"Total T: {t+1} Episode Num: {episode_num+1} SR_END : {avg_sr_end:.2f} SR_ONCE : {avg_sr_once:.2f} Reward: {avg_ret:.3f}")
            
            
            st_out_states = []
            #img_out_states = []
            out_actions = []
            out_rewards = []
            out_dones = []
            truncated = False
            
            state, _ = env.reset(seed=args.seed)
            st_state = state['state']
            img_state = state[f"{args.obs_mode}"]/255.0
            st_out_states.append( st_state )
            #img_out_states.append( img_state )
            st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
            #img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, h,w,c)

            episode_num += 1


        # Evaluate episode
        if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
            eval_counter += 1
            avg_reward, avg_sr_end, avg_sr_once = eval_transformer(policy, args, 1)
            experiment.add_scalar('Eval_reward', avg_reward, eval_counter)
            experiment.add_scalar('Eval_sr_end', avg_sr_end, eval_counter)
            experiment.add_scalar('Eval_sr_once', avg_sr_once, eval_counter)
                   
    torch.save(policy.trans, f"Maniskill_weights/{args.env}/[ST2]Trans|seed={args.seed}|ne={args.num_envs}|.pth")
    torch.save(policy.critic, f"Maniskill_weights/{args.env}/[ST2]Critic|seed={args.seed}|ne={args.num_envs}|.pth")


if __name__ == "__main__":

    with open("Shuttle/sh_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="PushCube-v1")          # OpenAI gym environment name
    parser.add_argument("--num_envs", default=5, type=int)
    parser.add_argument("--obs_mode", default='state')
    parser.add_argument("--seed", default=3, type=int)              # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=0, type=int)# 600
    parser.add_argument("--eval_freq", default=2e2, type=int)       # 2e2
    parser.add_argument("--max_timesteps", default=5e4, type=int)   # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)    # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=64, type=int)      # 600 for ICML
    parser.add_argument("--discount", default=0.8, type=float)     # 0.99
    parser.add_argument("--tau", default=0.01, type=float)         # 0.005
    parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()
    
    
    
    for seed in [3,3,3]:
        args.seed = seed
        path2run = f"RUNS/{args.env}/[ST2]|seed={args.seed}|ne={args.num_envs}"
    
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
            
        
    
        ######### ↓↓↓↓↓ БЛОК ДЛЯ ЗАГРУЗКИ ВЕСОВ ТРАНСФОРМЕРА ↓↓↓↓↓ ######### 
    
        pth2trans = f"Maniskill_weights/{args.env}/[ST1]Trans|seed={args.seed}|ne=40|.pth"
        pth2trans_tgt = f"Maniskill_weights/{args.env}/[ST1]Trans(t)|seed={args.seed}|ne=40|.pth"
        pth2critic = f"Maniskill_weights/{args.env}/[ST1]St_Critic|seed={args.seed}|ne=40|.pth"
        pth2critic_tgt = f"Maniskill_weights/{args.env}/[ST1]St_Critic(t)|seed={args.seed}|ne=40|.pth"
        
        kwargs["preload_weights"] = [pth2trans, pth2trans_tgt, pth2critic, pth2critic_tgt]
        policy = TD3(args.num_envs, 'rgb', config['train_config']['context_length'], config['model_config'], **kwargs)
        
        second_stage(policy, config, args, experiment)    