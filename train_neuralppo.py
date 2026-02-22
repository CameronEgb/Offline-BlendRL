import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


# added
from blendrl.agents.blender_agent import BlenderActorCritic
from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from nudge.utils import save_hyperparams
import os
import sys
import time
from pathlib import Path

import pickle
import random
import numpy as np
from rtpt import RTPT

from nudge.utils import load_model_train
from utils import CNNActor

# Log in to your W&B account
import wandb

OUT_PATH = Path("out/")
IN_PATH = Path("in/")

torch.set_num_threads(5)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = int(os.getenv("SEED", "1"))
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "NeuralPPO"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Seaquest-v4"
    """the id of the environment"""
    total_timesteps: int = int(os.getenv("ONLINE_STEPS", "0"))
    """total timesteps of the experiments"""
    num_envs: int = int(os.getenv("NUM_ENVS", "0"))
    """the number of parallel game environments"""
    num_steps: int = int(os.getenv("NUM_STEPS", "0"))
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = float(os.getenv("GAMMA", "0.99"))
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = int(os.getenv("PPO_EPOCHS", "4"))
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = float(os.getenv("ENT_COEF", "-1.0"))
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""
    
    # added
    env_name: str = os.getenv("ENVIRONMENT", "")
    """the name of the environment"""
    algorithm: str = os.getenv("ALGORITHM", "ppo")
    """the algorithm used in the agent"""
    blender_mode: str = os.getenv("BLENDER_MODE", "logic")
    """the mode for the blend"""
    blend_function: str = os.getenv("BLEND_FUNCTION", "softmax")
    """the function to blend the neural and logic agents: softmax or gumbel_softmax"""
    actor_mode: str = os.getenv("ACTOR_MODE", "hybrid")
    """the mode for the agent"""
    rules: str = os.getenv("RULES", "default")
    """the ruleset used in the agent"""
    save_steps: int = 5000000
    """the number of steps to save models"""
    pretrained: bool = False
    """to use pretrained neural agent"""
    joint_training: bool = False
    """jointly train neural actor and logic actor and blender"""
    learning_rate: float = float(os.getenv("LR", "0.0"))
    """the learning rate of the optimizer (neural)"""
    logic_learning_rate: float = float(os.getenv("LOGIC_LR", "0.0"))
    """the learning rate of the optimizer (logic)"""
    blender_learning_rate: float = float(os.getenv("BLENDER_LR", "0.0"))
    """the learning rate of the optimizer (blender)"""
    blend_ent_coef: float = float(os.getenv("BLEND_ENT_COEF", "-1.0"))
    """coefficient of the blend entropy"""
    recover: bool = False
    """recover the training from the last checkpoint"""
    reasoner: str = os.getenv("REASONER", "nsfr")
    """the reasoner used in the agent; nsfr or neumann"""
    
    # added for offline dataset generation
    save_dataset: bool = False
    """whether to save the dataset for offline training"""
    dataset_path: str = "offline_dataset"
    """path to save the dataset"""
    
    # added for orchestration
    run_id: str = ""
    """run id to override run name"""
    exp_id: str = os.getenv("EXPERIMENT_ID", "")
    """experiment ID for grouping runs"""
    intervals: int = int(os.getenv("INTERVALS_COUNT", "7"))
    """number of evaluation intervals (required)"""
    eval_episodes: int = int(os.getenv("EVAL_EPISODES", "100"))
    """number of evaluation episodes"""


def main():
        
    args = tyro.cli(Args)
    
    print(f"--- Hyperparameters Verified for {args.exp_id} ---")
    print(f"Total Steps: {args.total_timesteps} | Intervals: {args.intervals}")

    # Evaluation settings initialized early
    interval_results = []
    best_eval_reward = -float('inf')
    eval_step_freq = args.total_timesteps // (args.intervals - 1) if args.intervals > 1 else args.total_timesteps + 1

    rtpt = RTPT(
        name_initials="HS",
        experiment_name="AtariPPO",
        max_iterations=max(1, int(args.total_timesteps / args.save_steps)),
    )
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    import math
    args.num_iterations = math.ceil(args.total_timesteps / args.batch_size)
    
    if args.run_id:
        run_name = args.run_id
    else:
        run_name = f"{args.env_name}_{args.algorithm}_{args.seed}_{int(time.time())}"
        
    if args.track:
        wandb.init(
            project=args.wandb_project_name + "_" + args.env_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # for logging and model saving
    exp_subdir = args.exp_id
    
    experiment_dir = OUT_PATH / "runs" / exp_subdir / run_name 
    checkpoint_dir = experiment_dir / "checkpoints"
    writer_base_dir = OUT_PATH / "tensorboard" / exp_subdir
    writer_dir = writer_base_dir / run_name
    image_dir = experiment_dir / "images"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(writer_dir, exist_ok=True)
    
    # Dataset writer initialization
    dataset_writer = None
    if args.save_dataset:
        if args.env_name == "seaquest":
            from optimized_dataset_utils import SeaquestDatasetWriter
            if args.dataset_path == "offline_dataset":
                dataset_save_dir = experiment_dir / "offline_dataset"
            else:
                dataset_save_dir = Path(args.dataset_path) / exp_subdir / run_name
            dataset_writer = SeaquestDatasetWriter(save_dir=dataset_save_dir, env_name=args.env_name)
        else:
            from dataset_utils import DatasetWriter
            if args.dataset_path == "offline_dataset":
                # Save inside experiment directory
                dataset_save_dir = experiment_dir / "offline_dataset"
            else:
                # Save in separate large dataset directory, grouped by exp_id
                dataset_save_dir = Path(args.dataset_path) / exp_subdir / run_name
            dataset_writer = DatasetWriter(save_dir=dataset_save_dir, env_name=args.env_name)

    writer = SummaryWriter(writer_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = VectorizedNudgeBaseEnv.from_name(
        args.env_name, n_envs=args.num_envs, mode=args.algorithm, seed=args.seed
    )

    # The dataset contains raw action indices (0-17) for Atari.
    # For Mountain Car, we use the environment's actual actions.
    if args.env_name == "mountaincar":
        mlp_module_path = f"in/envs/{args.env_name}/mlp.py"
        from nsfr.utils.common import load_module
        module = load_module(mlp_module_path)
        agent = module.MLP(device=device, out_size=envs.n_actions).to(device)
    else:
        agent = CNNActor(n_actions=18).to(device)
        if args.pretrained:
            agent.load_state_dict(torch.load("models/neural_ppo_agent_Seaquest-v4.pth"))
            print("Pretrained neural agent loaded!!!")

    if args.recover:
        # load saved agent with the most recent step
        agent, most_recent_step = load_model_train(
            experiment_dir, n_envs=args.num_envs, device=device
        )
        # load training logs
        with open(checkpoint_dir / "training_log.pkl", "rb") as f:
            log_data = pickle.load(f)
            (
                episodic_returns,
                episodic_lengths,
                value_losses,
                policy_losses,
                entropies,
                blend_entropies,
            ) = log_data[:6]
            if len(log_data) >= 7:
                episodic_raw_returns = log_data[6]
            else:
                episodic_raw_returns = episodic_returns.copy()
        
        # Load interval results
        if (experiment_dir / "results.json").exists():
            with open(experiment_dir / "results.json", "r") as f:
                import json
                interval_results = json.load(f)
                if interval_results:
                    best_eval_reward = max([d["avg_reward"] for d in interval_results])
    else:
        episodic_returns = []
        episodic_raw_returns = [] # NEW: actual Atari score
        episodic_lengths = []
        value_losses = []
        policy_losses = []
        entropies = []
        blend_entropies = []
        
        # --- Step 0 Evaluation (Only if not recovering) ---
        print(f"--- Evaluating Interval 0 at Global Step 0 ---")
        n_eval_envs = 10
        eval_env = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=n_eval_envs, mode=args.algorithm, seed=args.seed + 100)
        eval_total_rewards = []
        eval_total_raw_rewards = []
        eval_cumulative_rewards = np.zeros(n_eval_envs)
        _, e_obs = eval_env.reset()
        e_obs = torch.Tensor(e_obs).to(device)
        
        while len(eval_total_rewards) < args.eval_episodes:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(e_obs)
            (e_logic_obs, e_obs), reward, terminations, truncations, infos = eval_env.step(action.cpu().numpy())
            e_obs = torch.Tensor(e_obs).to(device)
            
            # Manually track the SHAPED reward
            eval_cumulative_rewards += np.array(reward)
            
            for i in range(n_eval_envs):
                if terminations[i] or truncations[i]:
                    raw_reward = 0.0
                    if "final_info" in infos and infos["final_info"][i] is not None:
                        raw_reward = infos["final_info"][i].get("episode", {}).get("r", 0.0)
                    elif "episode" in infos and infos["_episode"][i]:
                        raw_reward = infos["episode"]["r"][i]
                    
                    eval_total_rewards.append(eval_cumulative_rewards[i])
                    eval_total_raw_rewards.append(raw_reward)
                    eval_cumulative_rewards[i] = 0 
                    
                    if len(eval_total_rewards) >= args.eval_episodes:
                        break
        
        avg_reward = np.mean(eval_total_rewards[:args.eval_episodes]) if eval_total_rewards else 0.0
        avg_raw_reward = np.mean(eval_total_raw_rewards[:args.eval_episodes]) if eval_total_raw_rewards else 0.0
        print(f"Interval 0 Eval Reward (Shaped): {avg_reward:.2f} | Raw: {avg_raw_reward:.2f}")
        writer.add_scalar("charts/eval_return", avg_reward, 0)
        writer.add_scalar("charts/eval_raw_return", avg_raw_reward, 0)
        interval_results.append({
            "interval": 0, 
            "data_limit": 0, 
            "avg_reward": float(avg_reward), 
            "avg_raw_reward": float(avg_raw_reward),
            "step": 0
        })
        eval_env.close()

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup moved after envs.reset() to support dynamic observation spaces

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    save_step_bar = 0  # args.save_steps
    episode_log_count = 0
    if args.recover:
        global_step = most_recent_step
        save_step_bar = most_recent_step
    start_time = time.time()
    next_logic_obs, next_obs = envs.reset()
    print("Environments reset and training started.")
    
    # ALGO Logic: Storage setup
    observation_space = next_obs.shape[1:]
    logic_observation_space = next_logic_obs.shape[1:]
    action_space = ()
    obs = torch.zeros((args.num_steps, args.num_envs) + observation_space).to(device)
    logic_obs = torch.zeros((args.num_steps, args.num_envs) + logic_observation_space).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + action_space).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    rtpt.start()

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        episodic_game_returns = torch.zeros((args.num_envs)).to(device)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            logic_obs[step] = next_logic_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            (real_next_logic_obs, real_next_obs), reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            real_next_logic_obs = real_next_logic_obs.float()
            
            terminations = np.array(terminations)
            truncations = np.array(truncations)
            real_next_done = np.logical_or(terminations, truncations)
            
            if dataset_writer is not None:
                # Cap the dataset at exactly args.total_timesteps
                current_count = global_step - args.num_envs
                if current_count < args.total_timesteps:
                    to_add = min(args.num_envs, args.total_timesteps - current_count)
                    if to_add == args.num_envs:
                        dataset_writer.batch_add(
                            obs=next_obs,
                            logic_obs=next_logic_obs,
                            action=action,
                            reward=reward,
                            next_obs=real_next_obs,
                            next_logic_obs=real_next_logic_obs,
                            done=real_next_done
                        )
                    else:
                        dataset_writer.batch_add(
                            obs=next_obs[:to_add],
                            logic_obs=next_logic_obs[:to_add],
                            action=action[:to_add],
                            reward=reward[:to_add],
                            next_obs=real_next_obs[:to_add],
                            next_logic_obs=real_next_logic_obs[:to_add],
                            done=real_next_done[:to_add]
                        )

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_logic_obs, next_done = (
                torch.Tensor(real_next_obs).to(device),
                torch.Tensor(real_next_logic_obs).to(device),
                torch.Tensor(real_next_done).to(device),
            )

            episodic_game_returns += torch.tensor(reward).to(device).view(-1)

            # --- Episode Logging Fix ---
            # Try multiple ways to detect completed episodes in vector environments
            found_episodes = []
            if "_episode" in infos:
                found_episodes = [k for k in range(args.num_envs) if infos["_episode"][k]]
            elif "episode" in infos and isinstance(infos["episode"], dict):
                # Some gymnasium versions or wrappers might not have _episode but have episode dict
                # If it's a dict with 'r', it's likely a VectorRecordEpisodeStatistics output
                if "r" in infos["episode"]:
                    found_episodes = [k for k in range(len(infos["episode"]["r"])) if infos["episode"]["r"][k] != 0 or infos["episode"]["l"][k] != 0]

            for k in found_episodes:
                episode_r = infos["episode"]["r"][k]
                episode_l = infos["episode"]["l"][k]
                
                if episode_log_count < 20:
                    print(f"env={k}, global_step={global_step}, episodic_return={episodic_game_returns[k].item()}, episodic_raw_return={episode_r}, episodic_length={episode_l}")
                writer.add_scalar("charts/episodic_return", episodic_game_returns[k], global_step)
                writer.add_scalar("charts/episodic_length", episode_l, global_step)
                episodic_returns.append(episodic_game_returns[k].item())
                episodic_raw_returns.append(episode_r)
                episodic_lengths.append(episode_l)
                episodic_game_returns[k] = 0
                episode_log_count += 1
                print(f"Iteration {iteration} | env {k} finished episode: return={episode_r}, length={episode_l}")
            # --- End Episode Logging Fix ---
              
        # Save training log every iteration (outside the step loop for efficiency)
        training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies, blend_entropies, episodic_raw_returns)
        with open(checkpoint_dir / "training_log.pkl", "wb") as f:
            pickle.dump(training_log, f)
                
        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + observation_space)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(newvalue - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs) if clipfracs else 0.0, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        
        if iteration % 1 == 0:
            sps = int(global_step / (time.time() - start_time))
            print(f"Iteration: {iteration}/{args.num_iterations}, Global Step: {global_step}, SPS: {sps}")
            writer.add_scalar("charts/SPS", sps, global_step)
            
        value_losses.append(v_loss.item())
        policy_losses.append(pg_loss.item())
        entropies.append(entropy_loss.item())
        
        # Interval evaluation - Strictly step-based trigger
        interval_idx = len(interval_results)
        if global_step >= (interval_idx * eval_step_freq) and interval_idx < args.intervals:
            print(f"--- Evaluating Interval {interval_idx} at Global Step {global_step} (Target: {interval_idx * eval_step_freq:.0f}) ---")
            
            n_eval_envs = 10
            eval_env = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=n_eval_envs, mode=args.algorithm, seed=args.seed + 100)
            eval_total_rewards = []
            eval_total_raw_rewards = []
            eval_cumulative_rewards = np.zeros(n_eval_envs)
            _, e_obs = eval_env.reset()
            e_obs = torch.Tensor(e_obs).to(device)
            
            while len(eval_total_rewards) < args.eval_episodes:
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(e_obs)
                (e_logic_obs, e_obs), reward, terminations, truncations, infos = eval_env.step(action.cpu().numpy())
                e_obs = torch.Tensor(e_obs).to(device)
                
                eval_cumulative_rewards += np.array(reward)
                
                for i in range(n_eval_envs):
                    if terminations[i] or truncations[i]:
                        raw_reward = 0.0
                        if "final_info" in infos and infos["final_info"][i] is not None:
                            raw_reward = infos["final_info"][i].get("episode", {}).get("r", 0.0)
                        elif "episode" in infos and infos["_episode"][i]:
                            raw_reward = infos["episode"]["r"][i]
                        
                        eval_total_rewards.append(eval_cumulative_rewards[i])
                        eval_total_raw_rewards.append(raw_reward)
                        eval_cumulative_rewards[i] = 0
                        
                        if len(eval_total_rewards) >= args.eval_episodes:
                            break
                
                if len(eval_total_rewards) >= args.eval_episodes:
                    break
            
            avg_reward = np.mean(eval_total_rewards[:args.eval_episodes]) if eval_total_rewards else 0.0
            avg_raw_reward = np.mean(eval_total_raw_rewards[:args.eval_episodes]) if eval_total_raw_rewards else 0.0
            print(f"Interval {interval_idx} Eval Reward (Shaped): {avg_reward:.2f} | Raw: {avg_raw_reward:.2f}")
            writer.add_scalar("charts/eval_return", avg_reward, global_step)
            writer.add_scalar("charts/eval_raw_return", avg_raw_reward, global_step)
            
            if avg_reward >= best_eval_reward:
                best_eval_reward = avg_reward
                checkpoint_path = checkpoint_dir / "best_model.pth"
                torch.save(agent.state_dict(), checkpoint_path)
                print(f"New best model saved with reward {avg_reward:.2f}")
                save_hyperparams(args=args, save_path=experiment_dir / "config.yaml", print_summary=False)
            
            interval_results.append({
                "interval": interval_idx,
                "data_limit": int(interval_idx * eval_step_freq),
                "avg_reward": float(avg_reward),
                "avg_raw_reward": float(avg_raw_reward),
                "step": global_step
            })
            with open(experiment_dir / "results.json", "w") as f:
                import json
                json.dump(interval_results, f, indent=4)
            
            training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies, blend_entropies, episodic_raw_returns)
            with open(checkpoint_dir / "training_log.pkl", "wb") as f:
                pickle.dump(training_log, f)
            eval_env.close()

    # Final Evaluation (if not already captured by the last interval)
    if len(interval_results) < args.intervals:
        interval_idx = len(interval_results)
        print(f"--- Final Evaluation at Global Step {global_step} ---")
        n_eval_envs = 10
        eval_env = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=n_eval_envs, mode=args.algorithm, seed=args.seed + 100)
        eval_total_rewards = []
        eval_total_raw_rewards = []
        eval_cumulative_rewards = np.zeros(n_eval_envs)
        _, e_obs = eval_env.reset()
        e_obs = torch.Tensor(e_obs).to(device)
        
        while len(eval_total_rewards) < args.eval_episodes:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(e_obs)
            (e_logic_obs, e_obs), reward, terminations, truncations, infos = eval_env.step(action.cpu().numpy())
            e_obs = torch.Tensor(e_obs).to(device)
            eval_cumulative_rewards += np.array(reward)
            for i in range(n_eval_envs):
                if terminations[i] or truncations[i]:
                    eval_total_rewards.append(eval_cumulative_rewards[i]); eval_cumulative_rewards[i] = 0
                    if "final_info" in infos and infos["final_info"][i] is not None:
                        eval_total_raw_rewards.append(infos["final_info"][i]["episode"]["r"])
                    elif "episode" in infos and infos["_episode"][i]:
                        eval_total_raw_rewards.append(infos["episode"]["r"][i])
                    if len(eval_total_rewards) >= args.eval_episodes: break
            if len(eval_total_rewards) >= args.eval_episodes: break
        
        avg_reward = np.mean(eval_total_rewards[:args.eval_episodes]) if eval_total_rewards else 0.0
        avg_raw_reward = np.mean(eval_total_raw_rewards[:args.eval_episodes]) if eval_total_raw_rewards else 0.0
        print(f"Final Eval Reward (Shaped): {avg_reward:.2f} | Raw: {avg_raw_reward:.2f}")
        writer.add_scalar("charts/eval_return", avg_reward, global_step)
        writer.add_scalar("charts/eval_raw_return", avg_raw_reward, global_step)
        
        interval_results.append({
            "interval": interval_idx, "data_limit": int(args.total_timesteps), 
            "avg_reward": float(avg_reward), "avg_raw_reward": float(avg_raw_reward), "step": global_step
        })
        with open(experiment_dir / "results.json", "w") as f: json.dump(interval_results, f, indent=4)
        eval_env.close()

    if dataset_writer is not None:
        dataset_writer.close()

    checkpoint_path = checkpoint_dir / "best_model.pth"
    torch.save(agent.state_dict(), checkpoint_path)
    print(f"Final agent has been saved to {checkpoint_path}")

    training_log = (episodic_returns, episodic_lengths, value_losses, policy_losses, entropies, blend_entropies, episodic_raw_returns)
    with open(checkpoint_dir / "training_log.pkl", "wb") as f:
        pickle.dump(training_log, f)

    envs.close()
    writer.close()
    
    end_time = time.time()
    duration = end_time - start_time
    import datetime
    runtime_data = {
        "runtime_seconds": duration,
        "runtime_formatted": str(datetime.timedelta(seconds=int(duration)))
    }
    with open(experiment_dir / "runtime.json", "w") as f:
        import json
        json.dump(runtime_data, f, indent=4)

if __name__ == "__main__":
    main()
