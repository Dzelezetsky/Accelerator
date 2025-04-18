import numpy as np
import torch
#import gym
import argparse
import os
import yaml
import copy

##########################################
from sh_tools4mujoco import TD3, New_Trans_RB, env_constructor, mean_padding, ReplayBuffer
##########################################


from torch.utils.tensorboard import SummaryWriter


device = "cuda" if torch.cuda.is_available() else "cpu"

with open("MUJOCO/sh_config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)




def eval_transformer(policy, args, eval_episodes=10):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, seed=args.seed+100, obs_indices=args.obs_indices)
    
    avg_reward = 0.

    policy.trans_actor.eval()

    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False
        
        st_state = state       #state (n_e, s_d)
        st_states = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #state (n_e, 1, s_d)
        
        t = -1
        while done == False:
            t+=1
            if st_states.shape[1] > policy.context_length :              # st_states: ne, cont, sd
                st_states = st_states[:, -policy.context_length:, :]
            
            st_s = st_states.unsqueeze(1)             #st_s (n_e, 1, cont, s_d)
            st_s = mean_padding(st_s, policy.context_length)
            
            sampled_action = policy.trans_actor.actor_forward(st_s)                   
            sampled_action = sampled_action.detach().cpu()
            action = np.clip( sampled_action.numpy()[:,0,], -1, 1)  #action (n_e, a_d)

            state, r, done, info = eval_env.step( action[0] )
            
            avg_reward += r.item()
            st_state = state#['state'] 
            st_cur_state = copy.deepcopy(st_state).unsqueeze(1).to(device=device, dtype=torch.float32)  #cur_state (n_e, 1, s_d)
            st_states = torch.cat([st_states, st_cur_state], dim=1)                                          #states (n_e, cont+1, s_d)

    avg_reward /= eval_episodes 
            
    print("---------------------------------------")
    print(f"Transormer evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    
    policy.trans_actor.train()
    
    return avg_reward

def eval_policy(policy, args, eval_episodes=10):
    
    eval_env, state_dim, action_dim = env_constructor(args.env, seed=args.seed+100, obs_indices=args.obs_indices)

    avg_reward = 0.
    
    out_states = []
    out_img_states = []
    out_actions = []
    
    policy.actor.eval()
    
    
    for _ in range(eval_episodes):
        state = eval_env.reset()
        done = False
        st_state = state              #state (n_e, s_d)
        out_states.append( st_state )
        
        t=0
        while done == False:
            t+=1
            action = policy.select_action(st_state)  # action: arr (n_e, a_d)
            
            out_actions.append(torch.Tensor(action))
            
            state, r, done, info = eval_env.step( action[0] ) #action[0] arr(,a_d)
            avg_reward += r.item()
            
            st_state = state               #state (n_e, s_d)
            out_states.append( st_state )
            
            #if (t > policy.context_length):
            states2RB = out_states[-policy.context_length-1:-1]
            next_states2RB = out_states[-policy.context_length:]
            act2RB = out_actions[-1]
            policy.trans_RB.recieve_traj(states2RB, next_states2RB, act2RB, r.reshape(1,-1), 1-done.to(int))
    
    avg_reward /= eval_episodes         
    

    print("---------------------------------------")
    print(f"MLP evaluation over {eval_episodes} episodes: {avg_reward:.3f} ")
    print("---------------------------------------")
    
    policy.actor.train()
    
    return avg_reward


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")                
    parser.add_argument("--env", default="HalfCheetah-v4")        
    parser.add_argument("--obs_indices", default=[0,1,2,3,8,9,10,11,12]) #Cth [0,1,2,3,8,9,10,11,12] | Hppr [0,1,2,3,4] | Ant [0,1,2,3,4,5,6,7,8,9,10,11,12]
    parser.add_argument("--obs_mode", default="state")
    parser.add_argument("--use_train_data", default=False)   # True, False
    parser.add_argument("--additional_ascent", default=None)  #MLP, Trans, None
    parser.add_argument("--additional_bellman", default=None)  #MLP, Trans, None
    parser.add_argument("--evals_for_trans", default=10)   # ???????
    parser.add_argument("--num_envs", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--trans_critic", default=False) # NOT USED
    parser.add_argument("--separate_trans_critic", default=False) # NOT USED
    parser.add_argument("--start_timesteps", default=25e3, type=int)
    parser.add_argument("--eval_freq", default=3e3, type=int)       # 2e3
    parser.add_argument("--max_timesteps", default=1.3e6, type=int) 
    parser.add_argument("--grad_clip", default=1000000, type=int)
    parser.add_argument("--expl_noise", default=0.1, type=float)    
    parser.add_argument("--batch_size", default=256, type=int)    
    parser.add_argument("--discount", default=0.99, type=float)     
    parser.add_argument("--tau", default=0.005, type=float)         
    parser.add_argument("--policy_noise", default=0.2)              
    parser.add_argument("--noise_clip", default=0.5)               
    parser.add_argument("--policy_freq", default=2, type=int)       
    parser.add_argument("--save_model", action="store_true")        
    parser.add_argument("--load_model", default="")                 
    args = parser.parse_args()

    
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
    print("---------------------------------------")

    with open("MUJOCO/sh_config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
        
    for RUN in [4]:  
        
        args.seed = RUN
        
        
        path2run = f"__RLW_MATERIALS/WORKSHOP_RUNS/{args.env}/[STAGE1_POMDP_LSTM]|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|EvFreq={args.eval_freq}"

        experiment = SummaryWriter(log_dir=path2run)
        
        env, state_dim, action_dim = env_constructor(args.env, seed=args.seed, obs_indices=args.obs_indices)
        
        env.action_space.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        max_action = 1.

        kwargs = {
            "state_dim": state_dim,
            "action_dim": action_dim,
            "max_action": max_action,
            "discount": args.discount,
            "tau": args.tau,
        }

        if args.policy == "TD3":
            
            kwargs["policy_noise"] = args.policy_noise * max_action
            kwargs["noise_clip"] = args.noise_clip * max_action
            kwargs["policy_freq"] = args.policy_freq
            kwargs["grad_clip"] = args.grad_clip
            policy = TD3(args.num_envs, args.obs_mode ,config['train_config']['context_length'], config['model_config'], **kwargs) 
            
        policy.experiment = experiment
        replay_buffer = ReplayBuffer(args.num_envs, state_dim, action_dim)

###################################
        vec_out_states = []
        out_actions = []
################################### 
        
        state = env.reset()#['state'] #tens(n_e, s_d)
        if args.use_train_data:
            vec_out_states.append(state)#####################################
        
        
        done = False
        episode_num = 0
        eval_counter = 0
        trans_eval_counter = 0
        avg_ret = 0.
        max_trans_reward = 0

        for t in range(int(args.max_timesteps)):
            

            # Select action randomly or according to policy
            if t < args.start_timesteps:
                action = env.action_space.sample()  # Array (, a_d)
            else:
                action = policy.select_action(state) # Array (n_e, a_d)
                noise = np.random.normal(0, args.expl_noise, size=action.shape) 
                action = (action + noise).clip(-max_action, max_action)[0]      # arr (, a_d)
            
               
            next_state, r, done, info = env.step( action )
#############################################################

            if args.use_train_data:
                out_actions.append(torch.Tensor(action))
                vec_out_states.append(next_state)
                
                #if (t > policy.context_length):
                vec_states2RB = vec_out_states[-policy.context_length-1:-1]
                next_vec_states2RB = vec_out_states[-policy.context_length:]
                act2RB = out_actions[-1]
                policy.trans_RB.recieve_traj(vec_states2RB, next_vec_states2RB, act2RB, r.reshape(1,-1), 1-done.to(int))

################################################################     
                   
            
            avg_ret += r.item()
            done_bool = done.to(float)

            # Store data in replay buffer
            replay_buffer.add(state.cpu(), action, next_state.cpu(), r.cpu(), done_bool.cpu())

            state = next_state

            # Train agent after collecting sufficient data
            if t >= args.start_timesteps:
                policy.train(replay_buffer, args.batch_size)
                if args.use_train_data:
                    policy.train_trans_actor(256, args.additional_ascent, args.additional_bellman)
            if done: 
                avg_ret = avg_ret / args.num_envs
                print(f"Total T: {t+1} Episode Num: {episode_num+1} | Reward: {avg_ret}")
                avg_ret = 0
                
                state = env.reset()
                state = state#['state']
                done = False
                episode_num += 1 

            # Evaluate episode
            if ((t + 1) % args.eval_freq == 0) and (t >= args.start_timesteps):
                eval_counter += 1
                avg_reward = eval_policy(policy, args, args.evals_for_trans)
                experiment.add_scalar('Eval_reward', avg_reward, t)
                
                policy.train_trans_actor(256, args.additional_ascent, args.additional_bellman)
                #policy.trans_RB.reset()
                tr_avg_reward = eval_transformer(policy, args, 1)
                experiment.add_scalar('Trans_Eval_reward_1', tr_avg_reward, t)
                
                if (tr_avg_reward > max_trans_reward) and (tr_avg_reward > 3000):
                    max_trans_reward = tr_avg_reward
                    torch.save(policy.trans_actor, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]Trans_actor|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth")
                    torch.save(policy.trans_actor_target, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]Trans_actor(t)|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth")
                    
                    torch.save(policy.trans_critic, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]Trans_critic|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth")
                    torch.save(policy.trans_critic_target, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]Trans_critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth")
                    
                    torch.save(policy.critic, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]St_Critic|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth")
                    torch.save(policy.critic_target, f"RLC_WEIGHTS/{args.env}/[FINAL_ST1(pomdp)]St_Critic(t)|seed={args.seed}|AddAsc={args.additional_ascent}|AddBell={args.additional_bellman}|UseTrData={args.use_train_data}|.pth") 
                
         