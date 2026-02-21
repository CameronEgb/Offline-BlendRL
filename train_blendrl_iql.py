import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

from blendrl.env_vectorized import VectorizedNudgeBaseEnv
from blendrl.agents.blender_agent import BlenderActorCritic
from dataset_utils import DatasetReader
from train_iql import QNetwork, ValueNetwork, MLPQNetwork, MLPValueNetwork

@dataclass
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = int(os.getenv("SEED", "1"))
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "offline_blendrl_iql"
    wandb_entity: str = None
    
    env_name: str = os.getenv("ENVIRONMENT", "")
    dataset_path: str = os.getenv("DATASET_PATH", "offline_dataset")
    dataset_run_name: str = ""
    
    total_timesteps: int = 1000000
    batch_size: int = int(os.getenv("BATCH_SIZE", "256"))
    learning_rate: float = float(os.getenv("OFFLINE_LR", "3e-4"))
    
    # BlendRL specific
    algorithm: str = os.getenv("ALGORITHM", "blender")
    blender_mode: str = os.getenv("BLENDER_MODE", "logic")
    blend_function: str = os.getenv("BLEND_FUNCTION", "softmax")
    actor_mode: str = os.getenv("ACTOR_MODE", "hybrid")
    rules: str = os.getenv("RULES", "default")
    reasoner: str = os.getenv("REASONER", "nsfr")
    
    # IQL params
    tau: float = float(os.getenv("IQL_TAU", "0.7"))
    beta: float = float(os.getenv("IQL_BETA", "3.0"))
    iql_tau: float = 0.005
    gamma: float = float(os.getenv("GAMMA", "0.99"))
    
    eval_freq: int = 5000
    eval_episodes: int = int(os.getenv("EVAL_EPISODES", "100"))
    
    # Interval training
    intervals: int = int(os.getenv("INTERVALS_COUNT", "7"))
    epochs_per_interval: int = int(os.getenv("OFFLINE_EPOCHS", "10"))
    
    # Orchestration
    run_id: str = ""
    exp_id: str = os.getenv("EXPERIMENT_ID", "")


def main():
    start_time = time.time()
    args = tyro.cli(Args)

    print(f"--- Hyperparameters Verified for {args.exp_id} ---")

    if args.run_id:
        run_name = args.run_id
    else:
        model_description = "{}_blender_{}".format(args.blend_function, args.blender_mode)
        run_name = f"BlendRLIQL_{args.env_name}_{model_description}_{args.seed}_{int(time.time())}"
    
    exp_subdir = args.exp_id
    experiment_dir = Path(f"out/runs/{exp_subdir}/{run_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        
    writer = SummaryWriter(str(experiment_dir))
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Load Dataset
    data_exp_id = args.exp_id
    if args.dataset_path == "offline_dataset":
        dataset_path = Path("out/runs") / data_exp_id / args.dataset_run_name / "offline_dataset"
    else:
        dataset_path = Path(args.dataset_path) / data_exp_id / args.dataset_run_name
    print(f"Loading dataset from {dataset_path}...")
    if args.env_name == "seaquest":
        from optimized_dataset_utils import SeaquestDatasetReader
        dataset = SeaquestDatasetReader(dataset_path, device=device)
    else:
        dataset = DatasetReader(dataset_path, device=device)
    total_transitions = len(dataset)
    print(f"Dataset loaded with {total_transitions} transitions.")
    
    if total_transitions == 0:
        print("Error: Dataset is empty. Exiting.")
        return

    envs = VectorizedNudgeBaseEnv.from_name(args.env_name, n_envs=1, mode=args.algorithm, seed=args.seed)

    agent = BlenderActorCritic(
        envs,
        args.rules,
        args.actor_mode,
        args.blender_mode,
        args.blend_function,
        args.reasoner,
        device,
    ).to(device)
    
    if args.env_name == "mountaincar":
        n_actions_iql = envs.n_actions
        _, temp_obs = envs.reset()
        num_in_features = temp_obs.shape[-1]
        q_network = MLPQNetwork(n_actions=n_actions_iql, num_in_features=num_in_features).to(device)
        q_network2 = MLPQNetwork(n_actions=n_actions_iql, num_in_features=num_in_features).to(device)
        target_q_network = MLPQNetwork(n_actions=n_actions_iql, num_in_features=num_in_features).to(device)
        target_q_network2 = MLPQNetwork(n_actions=n_actions_iql, num_in_features=num_in_features).to(device)
        value_network = MLPValueNetwork(num_in_features=num_in_features).to(device)
    else:
        n_actions_iql = 18
        q_network = QNetwork(n_actions=n_actions_iql).to(device)
        q_network2 = QNetwork(n_actions=n_actions_iql).to(device)
        target_q_network = QNetwork(n_actions=n_actions_iql).to(device)
        target_q_network2 = QNetwork(n_actions=n_actions_iql).to(device)
        value_network = ValueNetwork().to(device)
        
    target_q_network.load_state_dict(q_network.state_dict())
    target_q_network2.load_state_dict(q_network2.state_dict())
    
    optimizer_q = optim.Adam(list(q_network.parameters()) + list(q_network2.parameters()), lr=args.learning_rate)
    optimizer_value = optim.Adam(value_network.parameters(), lr=args.learning_rate)
    optimizer_actor = optim.Adam(
        [
            {"params": agent.visual_neural_actor.parameters(), "lr": args.learning_rate},
            {"params": agent.logic_actor.parameters(), "lr": args.learning_rate},
            {"params": agent.blender.parameters(), "lr": args.learning_rate},
        ],
        lr=args.learning_rate
    )

    print("Starting training...")
    global_step = 0
    interval_results = []
    best_eval_reward = -float('inf')
    
    if args.intervals > 1:
        interval_step_size = total_transitions // (args.intervals - 1)
    else:
        interval_step_size = total_transitions

    for interval in range(0, args.intervals):
        current_limit = min(interval * interval_step_size, total_transitions)
        if interval == args.intervals - 1:
            current_limit = total_transitions
            
        dataset.set_limit(current_limit)
        steps_per_interval = int((current_limit / args.batch_size) * args.epochs_per_interval) if current_limit > 0 else 0
        steps_per_interval = max(steps_per_interval, 1) if current_limit > 0 else 0
        
        print(f"--- Interval {interval}/{args.intervals - 1} | Limit: {current_limit} | Steps: {steps_per_interval} ---")
        
        for step in range(steps_per_interval):
            global_step += 1
            batch = dataset.sample(args.batch_size)
            obs, logic_obs, actions, rewards, next_obs, dones = batch["obs"], batch["logic_obs"], batch["action"], batch["reward"], batch["next_obs"], batch["done"]

            if args.env_name == "mountaincar":
                shaping = torch.abs(next_obs[:, 1]) * 100.0
                rewards = rewards + shaping

            with torch.no_grad():
                next_v = value_network(next_obs).view(-1)
                q_target = rewards + args.gamma * next_v * (1 - dones)
                
            current_q1_a = q_network(obs).gather(1, actions.unsqueeze(1)).view(-1)
            current_q2_a = q_network2(obs).gather(1, actions.unsqueeze(1)).view(-1)
            q_loss = F.mse_loss(current_q1_a, q_target) + F.mse_loss(current_q2_a, q_target)
            optimizer_q.zero_grad(); q_loss.backward(); optimizer_q.step()
            
            with torch.no_grad():
                t_q_a = torch.min(target_q_network(obs), target_q_network2(obs)).gather(1, actions.unsqueeze(1)).view(-1)
            value = value_network(obs).view(-1)
            diff = t_q_a - value
            weight = torch.where(diff > 0, args.tau, 1 - args.tau)
            value_loss = (weight * (diff**2)).mean()
            optimizer_value.zero_grad(); value_loss.backward(); optimizer_value.step()
            
            with torch.no_grad():
                weights = torch.clamp(torch.exp(args.beta * (t_q_a - value)), max=100.0)
            _, log_probs, _, _, _ = agent.get_action_and_value(obs, logic_obs, actions)
            actor_loss = -(weights * log_probs).mean()
            optimizer_actor.zero_grad(); actor_loss.backward(); optimizer_actor.step()
            
            for param, target_param in zip(q_network.parameters(), target_q_network.parameters()):
                target_param.data.copy_(args.iql_tau * param.data + (1 - args.iql_tau) * target_param.data)
            for param, target_param in zip(q_network2.parameters(), target_q_network2.parameters()):
                target_param.data.copy_(args.iql_tau * param.data + (1 - args.iql_tau) * target_param.data)
                
            if global_step % 1000 == 0:
                writer.add_scalar("losses/q_loss", q_loss.item(), global_step)
                writer.add_scalar("losses/value_loss", value_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)

        print(f"Evaluating after Interval {interval}...")
        total_rewards = []
        next_logic_obs, next_obs = envs.reset()
        next_obs, next_logic_obs = torch.Tensor(next_obs).to(device), next_logic_obs.to(device)
        current_rewards = np.zeros(1)
        episodes_completed = 0
        while episodes_completed < args.eval_episodes:
            with torch.no_grad():
                action, _, _, _, _ = agent.get_action_and_value(next_obs, next_logic_obs)
            (next_logic_obs, next_obs), reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_obs, next_logic_obs = torch.Tensor(next_obs).to(device), torch.Tensor(next_logic_obs).to(device)
            current_rewards += reward
            if terminations[0] or truncations[0]:
                total_rewards.append(current_rewards[0])
                current_rewards = np.zeros(1); episodes_completed += 1
                next_logic_obs, next_obs = envs.reset()
                next_obs, next_logic_obs = torch.Tensor(next_obs).to(device), next_logic_obs.to(device)
        
        avg_reward = np.mean(total_rewards)
        print(f"Interval {interval} Eval Reward: {avg_reward}")
        writer.add_scalar("charts/eval_return", avg_reward, global_step)
        save_path = experiment_dir / "checkpoints"
        save_path.mkdir(parents=True, exist_ok=True)
        if avg_reward >= best_eval_reward:
            best_eval_reward = avg_reward
            agent.save(save_path / "best_model.pth", save_path, [], [], [])
            print(f"New best model saved with reward {avg_reward:.2f}")
        interval_results.append({"interval": interval, "data_limit": current_limit, "avg_reward": avg_reward, "step": global_step})
        with open(experiment_dir / "results.json", "w") as f: json.dump(interval_results, f, indent=4)

    agent.save(save_path / "best_model.pth", save_path, [], [], [])
    envs.close(); writer.close()
    end_time = time.time(); duration = end_time - start_time
    import datetime
    runtime_data = {"runtime_seconds": duration, "runtime_formatted": str(datetime.timedelta(seconds=int(duration)))}
    with open(experiment_dir / "runtime.json", "w") as f: json.dump(runtime_data, f, indent=4)

if __name__ == "__main__":
    main()
