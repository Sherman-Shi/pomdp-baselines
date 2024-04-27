import gym
import numpy as np
import torch
import torchkit.pytorch_utils as ptu
import wandb 

# import environments
import envs.pomdp

# import recurrent model-free RL (separate architecture)
from policies.models.policy_rnn import ModelFreeOffPolicy_Separate_RNN as Policy_RNN

# import the replay buffer
from buffers.seq_replay_buffer_vanilla import SeqReplayBuffer
from utils import helpers as utl

# Initialize wandb
wandb.init(project='pomdp_dataset_baselines', entity='sonnol')

# Define all arguments
args = {
    'cuda_id': 1,  # -1 if using cpu
    'env_name': "Hopper-v3",
    'algo_name': 'sac',
    'rnn_name': 'gru',
    'num_updates_per_iter': 5.0,
    'sampled_seq_len': 64,
    'buffer_size': 1e4,
    'batch_size': 32,
    'num_iters': 300,
    'num_init_rollouts_pool': 1,
    'num_rollouts_per_iter': 1,
    'eval_num_rollouts': 10,
    'log_interval': 20,
}

# Initialize wandb config
wandb.config.update(args)

# Set GPU mode
ptu.set_gpu_mode(torch.cuda.is_available() and args['cuda_id'] >= 0, args['cuda_id'])

# Initialize environment
env = gym.make(args['env_name'])
max_trajectory_len = env._max_episode_steps
act_dim = env.action_space.shape[0]
obs_dim = env.observation_space.shape[0]
# Initialize agent
agent_args = {
    'obs_dim': obs_dim,
    'action_dim': act_dim,
    'encoder': args['rnn_name'],
    'algo_name': args['algo_name'],
    'action_embedding_size': 1,
    'observ_embedding_size': 256,
    'reward_embedding_size': 1,
    'rnn_hidden_size': 1,
    'dqn_layers': [256, 256],
    'policy_layers': [256, 256],
    'lr': 0.0003,
    'gamma': 0.9,
    'tau': 0.005,
}
agent = Policy_RNN(**agent_args).to(ptu.device)
wandb.config.update(agent_args)

# Initialize policy storage buffer
policy_storage = SeqReplayBuffer(
    max_replay_buffer_size=int(args['buffer_size']),
    observation_dim=obs_dim,
    action_dim=act_dim,
    sampled_seq_len=args['sampled_seq_len'],
    sample_weight_baseline=0.0,
)

# Define collect_rollouts function
@torch.no_grad()
def collect_rollouts(
    num_rollouts, random_actions=False, deterministic=False, train_mode=True
):
    total_steps = 0
    total_rewards = 0.0

    for idx in range(num_rollouts):
        steps = 0
        rewards = 0.0
        obs = ptu.from_numpy(env.reset())
        obs = obs.reshape(1, obs.shape[-1])
        done_rollout = False

        action, reward, internal_state = agent.get_initial_info()

        if train_mode:
            obs_list, act_list, rew_list, next_obs_list, term_list = ([], [], [], [], [])

        while not done_rollout:
            if random_actions:
                action = ptu.FloatTensor([env.action_space.sample()])
            else:
                (action, _, _, _), internal_state = agent.act(
                    prev_internal_state=internal_state,
                    prev_action=action,
                    reward=reward,
                    obs=obs,
                    deterministic=deterministic,
                )

            next_obs, reward, done, info = utl.env_step(env, action.squeeze(dim=0))
            done_rollout = False if ptu.get_numpy(done[0][0]) == 0.0 else True

            steps += 1
            rewards += reward.item()

            term = (
                False
                if "TimeLimit.truncated" in info or steps >= max_trajectory_len
                else done_rollout
            )

            if train_mode:
                obs_list.append(obs)
                act_list.append(action)
                rew_list.append(reward)
                term_list.append(term)
                next_obs_list.append(next_obs)

            obs = next_obs.clone()

        if train_mode:
            policy_storage.add_episode(
                observations=ptu.get_numpy(torch.cat(obs_list, dim=0)),
                actions=ptu.get_numpy(torch.cat(act_list, dim=0)),
                rewards=ptu.get_numpy(torch.cat(rew_list, dim=0)),
                terminals=np.array(term_list).reshape(-1, 1),
                next_observations=ptu.get_numpy(torch.cat(next_obs_list, dim=0)),
            )

        total_steps += steps
        total_rewards += rewards

    if train_mode:
        return total_steps, total_rewards / num_rollouts
    else:
        return total_rewards / num_rollouts

# Define update function
def update(num_updates):
    rl_losses_agg = {}
    for update in range(num_updates):
        batch = ptu.np_to_pytorch_batch(policy_storage.random_episodes(args['batch_size']))
        rl_losses = agent.update(batch)
        for k, v in rl_losses.items():
            if update == 0:
                rl_losses_agg[k] = [v]
            else:
                rl_losses_agg[k].append(v)
    for k in rl_losses_agg:
        rl_losses_agg[k] = np.mean(rl_losses_agg[k])
    return rl_losses_agg

total_rollouts = args['num_init_rollouts_pool'] + args['num_iters'] * args['num_rollouts_per_iter']
n_env_steps_total = max_trajectory_len * total_rollouts
_n_env_steps_total = 0

# Initial rollouts
env_steps, training_return = collect_rollouts(
    num_rollouts=args['num_init_rollouts_pool'], random_actions=True, train_mode=True
)
_n_env_steps_total += env_steps

# Training loop
last_eval_num_iters = 0
learning_curve = {"x": [], "y": []}
while _n_env_steps_total < n_env_steps_total:
    env_steps, training_return = collect_rollouts(num_rollouts=args['num_rollouts_per_iter'], train_mode=True)
    _n_env_steps_total += env_steps
    
    train_stats = update(int(args['num_updates_per_iter'] * env_steps))

    current_num_iters = _n_env_steps_total // (args['num_rollouts_per_iter'] * max_trajectory_len)
    wandb.log({'global_env_steps': _n_env_steps_total, 'training_return': training_return})

    if current_num_iters != last_eval_num_iters and current_num_iters % args['log_interval'] == 0:
        last_eval_num_iters = current_num_iters
        eval_returns = collect_rollouts(
            num_rollouts=args['eval_num_rollouts'],
            train_mode=False,
            random_actions=False,
            deterministic=True,
        )
        learning_curve["x"].append(_n_env_steps_total)
        learning_curve["y"].append(eval_returns)
        print(_n_env_steps_total, eval_returns)
        wandb.log({'current_num_iters': current_num_iters, 'eval_returns': eval_returns})

wandb.finish()
