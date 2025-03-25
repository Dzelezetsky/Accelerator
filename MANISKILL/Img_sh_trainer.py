import numpy as np
import torch
#import gym
import argparse
import os
import yaml
from sh_tools4maniskill import TD3, ReplayBuffer, env_constructor
import copy

##########################################
from collections import defaultdict

import mani_skill.envs
import gymnasium as gym
#from mani_skill.utils.wrappers import CPUGymWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from collections import defaultdict
##########################################


from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("MANISKILL/image/config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)



def img_eval_transformer(policy, args, eval_episodes=1):
    '''
    
    Transformers evaluation based on images and vector states.
    Agent recieves tensors (n_e, 1, context, 128, 128, 3/4) and (n_e, 1, context, s_d)
    ----- and returns tensor (n_e, 1, a_d)
    
    '''
    eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'rgb', reconf_freq=1)
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    policy.trans_actor.eval()
    policy.trans_critic.eval()
    for _ in range(eval_episodes):
        state, info = eval_env.reset(seed=args.seed) 
        img_state = state['rgb']/255.0            #img_state (n_e, 128, 128, 4/3)
        vec_state = state['state']            #img_state (n_e, s_d)
        img_states = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #states (n_e, 1, 128, 128, 4/3)
        vec_states = copy.deepcopy(vec_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #states (n_e, 1, s_d)
        
        t = -1
        while "final_info" not in info:
            t+=1
            if vec_states.shape[1] > policy.context_length :
                img_states = img_states[:, -policy.context_length:, :]
                vec_states = vec_states[:, -policy.context_length:, :]

            img_s = img_states.unsqueeze(1)              #s (n_e, 1, cont, 128, 128, 4/3)
            vec_s = vec_states.unsqueeze(1)              #s (n_e, 1, cont, s_d)
            sampled_action = policy.trans_actor.actor_forward(vec_s, img_s)  # sampled_action Tens(n_e, 1, a_d)
            sampled_action = sampled_action.detach().cpu()
            action = np.clip( sampled_action.numpy()[:,0,], -1, 1)  #action Arr(n_e, a_d)

            state, r, terminated, truncated, info = eval_env.step( action )
            
            img_state = state['rgb']/255.0            
            vec_state = state['state']           
            img_cur_state = copy.deepcopy(img_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #img_cur_state (n_e, 1, 128, 128, 4/3)
            vec_cur_state = copy.deepcopy(vec_state).unsqueeze(1).to(device=device, dtype=torch.float32)
            vec_states = torch.cat([vec_states, vec_cur_state], dim=1)
            img_states = torch.cat([img_states, img_cur_state], dim=1)                                          #img_states (n_e, cont, 128, 128, 4/3)

        avg_sr_once += info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
        avg_sr_end += info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
        avg_reward += info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs 
            

    avg_reward /= eval_episodes
    avg_sr_end /= eval_episodes
    avg_sr_once /= eval_episodes

    print("---------------------------------------")
    print(f"Seed {args.seed} | Transformer Evaluation over {eval_episodes} episodes: {avg_reward:.3f} ||| {avg_sr_end:.3f} ||| {avg_sr_once:.3f}")
    print("---------------------------------------")
    policy.trans_actor.train()
    policy.trans_critic.train()
    return avg_reward, avg_sr_end, avg_sr_once

def img_eval_policy(policy, args, eval_episodes=1):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'rgb', reconf_freq=1)
    #sim_env, _, _ = env_constructor(args.env, args.num_envs, 'rgb', reconf_freq=1)
    avg_reward = 0.
    avg_sr_end = 0.
    avg_sr_once= 0.
    
    #full_vec_out_states = []
    img_out_states = []
    vec_out_states = []
    out_actions = []
    
    for _ in range(eval_episodes):
        state, info = eval_env.reset(seed=args.seed)
        vec_state = state['state']
        img_state = state['rgb']
        #sim_state, _ = sim_env.reset(seed=args.seed)
        #img_sim_state = sim_state['rgb'] # (n_e, 1, 128, 128, 4/3)
        #vec_sim_state = sim_state['state'] # (n_e, 1, s_d)
        img_out_states.append( img_state )
        vec_out_states.append( vec_state )
        
        t=0
        
        while "final_info" not in info:
            t+=1
            action = policy.select_action(img_state/255.0, vec_state)  # action: arr (n_e, a_d)
            
            out_actions.append(torch.Tensor(action))  # Tens (n_e, a_d)
            
            state, r, terminated, truncated, info = eval_env.step(action)
            vec_state = state['state'] # (n_e, 1, 128, 128, 4/3)
            img_state = state['rgb']  # (n_e, 1, s_d)
            
            img_out_states.append( img_state )
            vec_out_states.append( vec_state )
            
            if (t >= policy.context_length):
                #full_vec_states2RB = full_vec_out_states[-2]
                img_states2RB = img_out_states[-policy.context_length-1:-1]
                vec_states2RB = vec_out_states[-policy.context_length-1:-1]
                act2RB = out_actions[-1]
                policy.trans_RB.recieve_traj(img_states2RB, vec_states2RB, act2RB)
            
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
    parser.add_argument("--policy", default="TD3")                  # 
    parser.add_argument("--env", default="PushCube-v1")          # 
    parser.add_argument("--num_envs", default=12, type=int)
    parser.add_argument("--seed", default=2, type=int)
    parser.add_argument("--use_train_data", default=False, type=bool)
    parser.add_argument("--additional_ascent", default=None, type=bool)  
    parser.add_argument("--start_trans_train", default=0, type=int)  #30
    parser.add_argument("--start_timesteps", default=5e2, type=int)# 5e2
    parser.add_argument("--eval_freq", default=70000, type=int)       # 2e2 6e2
    parser.add_argument("--max_timesteps", default=70000, type=int)   # !!!!!50000!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    parser.add_argument("--expl_noise", default=0.1, type=float)    # 
    parser.add_argument("--batch_size", default=32, type=int)      # 256
    parser.add_argument("--discount", default=0.8, type=float)     # 0.99
    parser.add_argument("--tau", default=0.01, type=float)         # 0.005
    parser.add_argument("--policy_noise", default=0.2)              # 
    parser.add_argument("--noise_clip", default=0.5)                # R
    parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    file_name = f"{args.policy}_{args.env}_{args.seed}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    
    for RUN in [2,3]:
        args.seed = RUN
        
        path2run = f"WORKSHOP_RUNS/{args.env}/[NEW_ST_ST1]|seed={args.seed}||UseTrData={args.use_train_data}|"
        experiment = SummaryWriter(log_dir=path2run)
        
        env, state_dim, action_dim = env_constructor(args.env, args.num_envs, 'rgb', reconf_freq=None)

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

        # Initialize policy
        if args.policy == "TD3":
            # Target policy smoothing is scaled wrt the action scale
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            kwargs["preload_weights"] = None
            policy = TD3(args.num_envs, 'state', config['train_config']['context_length'], config['model_config'], **kwargs)
        
        policy.experiment = experiment
        replay_buffer = ReplayBuffer(args.num_envs, state_dim, action_dim, max_size=7000)
        
###################################
        img_out_states = []
        vec_out_states = []
        out_actions = []
###################################      
        
        state, _ = env.reset(seed=args.seed)  #tensor n_e, s_d
        vec_state = state['state']
        img_state = state['rgb']
        if args.use_train_data:
            img_out_states.append(img_state)
            vec_out_states.append(vec_state)#####################################
        
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
                action = policy.select_action(img_state/255.0, vec_state)
                noise = np.random.normal(0, args.expl_noise, size=action.shape) 
                action = (action + noise).clip(-max_action, max_action)      # arr n_e, a_d
                
                # action = (
                #     np.array( policy.select_action(state) )
                #     + np.random.normal(0, args.expl_noise, size=action_dim)
                # ).clip(-max_action, max_action)

            
            
            next_state, reward, terminated, truncated, info = env.step(action)
            img_next_state = next_state['rgb']
            vec_next_state = next_state['state']
            done_bool = truncated.to(float)
            
#############################################################
            if args.use_train_data:
                img_out_states.append(img_next_state)
                vec_out_states.append(vec_next_state)
                
                out_actions.append(torch.Tensor(action))################################
                if (t >= policy.context_length):
                    img_states2RB = img_out_states[-policy.context_length-1:-1]
                    vec_states2RB = vec_out_states[-policy.context_length-1:-1]
                    act2RB = out_actions[-1]
                    policy.trans_RB.recieve_traj(img_states2RB, vec_states2RB, act2RB)

################################################################            

            # Store data in replay buffer
            replay_buffer.add(img_state.cpu(), vec_state.cpu(), action, img_next_state.cpu(), vec_next_state.cpu(), reward.cpu(), done_bool.cpu())

            img_state = img_next_state
            vec_state = vec_next_state
            
            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
                if args.use_train_data:
                    policy.IMG_train_trans(policy, 32, experiment, additional_ascent=args.additional_ascent)
            if "final_info" in info: 
                avg_sr_once = info['final_info']['episode']['success_once'].cpu().numpy().sum() / args.num_envs
                avg_sr_end = info['final_info']['episode']['success_at_end'].cpu().numpy().sum() / args.num_envs
                avg_ret = info['final_info']['episode']['return'].cpu().numpy().sum() / args.num_envs
                
                print(f"Total T: {t+1} Episode Num: {episode_num+1} SR_END : {avg_sr_end:.2f} SR_ONCE : {avg_sr_once:.2f} Reward: {avg_ret:.3f}")
                
                state, _ = env.reset(seed=args.seed)
                img_state = state['rgb']
                vec_state = state['state']
                truncated = False
                episode_num += 1 

            # Evaluate episode
            if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
                eval_counter += 1
                avg_reward, avg_sr_end, avg_sr_once = img_eval_policy(policy, args, 10)
                tr_avg_reward, tr_avg_sr_end, tr_avg_sr_once = img_eval_transformer(policy, args, 1)
                
                experiment.add_scalar('Eval_reward', avg_reward, t)
                experiment.add_scalar('Eval_sr_end', avg_sr_end, t)
                experiment.add_scalar('Eval_sr_once', avg_sr_once, t)
                
                experiment.add_scalar('Trans_Eval_reward', tr_avg_reward, t)
                experiment.add_scalar('Trans_Eval_sr_end', tr_avg_sr_end, t)
                experiment.add_scalar('Trans_Eval_sr_once', tr_avg_sr_once, t)
                    
                if avg_reward > args.start_trans_train:
                    args.eval_freq = 200
                    policy.Img_train_trans(policy, 32, experiment, additional_ascent=args.additional_ascent) #256
                    
                    
                    # if (tr_avg_reward > max_trans_reward) and (tr_avg_reward > 10):
                    #     max_trans_reward = tr_avg_reward
                    #     torch.save(policy.trans, f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]Trans|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                    #     torch.save(policy.trans_target, f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]Trans(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                    #     torch.save(policy.critic, f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]St_Critic|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth")
                    #     torch.save(policy.critic_target, f"WORKSHOP_WEIGHTS/{args.env}/[NEW_ST_ST1]St_Critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|UseTrData={args.use_train_data}|.pth") 
                else:
                    policy.trans_RB.reset()    
                
                
                
            
              
        
        # torch.save(policy.trans, f"Maniskill_weights/{args.env}/[ST_ST1]Trans|seed={args.seed}|ne={args.num_envs}|.pth")
        # torch.save(policy.trans_target, f"Maniskill_weights/{args.env}/[ST_ST1]Trans(t)|seed={args.seed}|ne={args.num_envs}|.pth")
                        
        # torch.save(policy.critic, f"Maniskill_weights/{args.env}/[ST_ST1]Critic|seed={args.seed}|ne={args.num_envs}|.pth")
        # torch.save(policy.critic_target, f"Maniskill_weights/{args.env}/[ST_ST1]Critic(t)|seed={args.seed}|ne={args.num_envs}|.pth")
        
           