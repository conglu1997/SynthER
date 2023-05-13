# Train diffusion model on D4RL transitions.
import argparse
import pathlib

import d4rl
import gin
import gym
import numpy as np
import torch
import wandb

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import make_inputs, split_diffusion_samples, construct_diffusion_model


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    # wandb config
    parser.add_argument('--wandb-project', type=str, default="offline-rl-diffusion")
    parser.add_argument('--wandb-entity', type=str, default="")
    parser.add_argument('--wandb-group', type=str, default="diffusion_training")
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--save_num_samples', type=int, default=int(5e6))
    parser.add_argument('--save_file_name', type=str, default='5m_samples.npz')
    parser.add_argument('--load_checkpoint', action='store_true')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    env = gym.make(args.dataset)
    inputs = make_inputs(env)
    inputs = torch.from_numpy(inputs).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(inputs=inputs)
    trainer = Trainer(
        diffusion,
        dataset,
        results_folder=args.results_folder,
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=args.results_folder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        trainer.ema.to(trainer.accelerator.device)
        # Load the last checkpoint.
        trainer.load(milestone=trainer.train_num_steps)

    # Generate samples and save them.
    if args.save_samples:
        generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=trainer.ema.ema_model,
        )
        observations, actions, rewards, next_observations, terminals = generator.sample(
            num_samples=args.save_num_samples,
        )
        np.savez_compressed(
            results_folder / args.save_file_name,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )
