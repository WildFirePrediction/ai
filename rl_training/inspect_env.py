"""
Quick script to inspect environment data quality
"""
import pickle
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rl_training.wildfire_env_spatial import WildfireEnvSpatial

env_file = Path('//tilling_data/environments/env_00000.pkl')

# Load raw data
with open(env_file, 'rb') as f:
    env_data = pickle.load(f)

print("=" * 60)
print("ENVIRONMENT DATA INSPECTION")
print("=" * 60)

print(f"\nKeys in env_data: {list(env_data.keys())}")
print(f"\nMetadata:")
for k, v in env_data['metadata'].items():
    print(f"  {k}: {v}")

# Check static features
static = env_data['static']
print(f"\nStatic features keys: {list(static.keys())}")
print(f"  continuous shape: {static['continuous'].shape}")
print(f"  continuous min: {static['continuous'].min():.4f}, max: {static['continuous'].max():.4f}")
print(f"  continuous has NaN: {np.isnan(static['continuous']).any()}")
print(f"  continuous has Inf: {np.isinf(static['continuous']).any()}")
print(f"  lcm shape: {static['lcm'].shape}, min: {static['lcm'].min()}, max: {static['lcm'].max()}")
print(f"  fsm shape: {static['fsm'].shape}, min: {static['fsm'].min()}, max: {static['fsm'].max()}")

# Check temporal data
temporal = env_data['temporal']
print(f"\nTemporal features keys: {list(temporal.keys())}")
fire_masks = temporal['fire_masks']
fire_intensities = temporal['fire_intensities']
fire_temps = temporal['fire_temps']
fire_ages = temporal['fire_ages']
weather = temporal['weather_states']

print(f"  fire_masks shape: {fire_masks.shape}")
print(f"  fire_intensities shape: {fire_intensities.shape}")
print(f"    min: {fire_intensities.min():.4f}, max: {fire_intensities.max():.4f}")
print(f"    has NaN: {np.isnan(fire_intensities).any()}")
print(f"    has Inf: {np.isinf(fire_intensities).any()}")
print(f"  fire_temps shape: {fire_temps.shape}")
print(f"    min: {fire_temps.min():.4f}, max: {fire_temps.max():.4f}")
print(f"    has NaN: {np.isnan(fire_temps).any()}")
print(f"    has Inf: {np.isinf(fire_temps).any()}")
print(f"  fire_ages shape: {fire_ages.shape}")
print(f"    min: {fire_ages.min():.4f}, max: {fire_ages.max():.4f}")
print(f"  weather_states shape: {weather.shape}")
print(f"    min: {weather.min():.4f}, max: {weather.max():.4f}")
print(f"    has NaN: {np.isnan(weather).any()}")
print(f"    has Inf: {np.isinf(weather).any()}")

# Now check observations and rewards by simulating an episode
print("\n" + "=" * 60)
print("SIMULATING EPISODE TO CHECK REWARDS")
print("=" * 60)

env = WildfireEnvSpatial(env_file)
obs, info = env.reset()

print(f"\nInitial observation shape: {obs.shape}")
print(f"  Min: {obs.min():.4f}, Max: {obs.max():.4f}")
print(f"  Has NaN: {np.isnan(obs).any()}")
print(f"  Has Inf: {np.isinf(obs).any()}")

# Collect rewards by taking random actions
rewards = []
for _ in range(env.T - 1):
    action = np.random.randint(0, 9)
    obs, reward, done, info = env.step(action)
    rewards.append(reward)
    if done:
        break

rewards = np.array(rewards)
print(f"\nRewards collected: {len(rewards)}")
print(f"  Min: {rewards.min():.4f}")
print(f"  Max: {rewards.max():.4f}")
print(f"  Mean: {rewards.mean():.4f}")
print(f"  Std: {rewards.std():.4f}")
print(f"  Sum: {rewards.sum():.4f}")
print(f"  Has NaN: {np.isnan(rewards).any()}")

print(f"\n  Reward distribution:")
print(f"    < -1: {(rewards < -1).sum()}")
print(f"    -1 to -0.1: {((rewards >= -1) & (rewards < -0.1)).sum()}")
print(f"    -0.1 to 0: {((rewards >= -0.1) & (rewards < 0)).sum()}")
print(f"    0 to 0.1: {((rewards >= 0) & (rewards < 0.1)).sum()}")
print(f"    0.1 to 1: {((rewards >= 0.1) & (rewards < 1)).sum()}")
print(f"    > 1: {(rewards >= 1).sum()}")

# Check a few more environments
print("\n" + "=" * 60)
print("CHECKING 10 MORE ENVIRONMENTS")
print("=" * 60)

all_rewards = []
all_obs_stats = []
for i in range(1, 11):
    env_file = Path(f'//tilling_data/environments/env_{i:05d}.pkl')
    env = WildfireEnvSpatial(env_file)
    obs, _ = env.reset()

    # Check for data issues
    if np.isnan(obs).any() or np.isinf(obs).any():
        print(f"WARNING: env_{i:05d}.pkl has NaN or Inf in observations!")

    all_obs_stats.append({
        'min': obs.min(),
        'max': obs.max(),
        'mean': obs.mean()
    })

    # Collect rewards
    for _ in range(env.T - 1):
        action = np.random.randint(0, 9)
        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)
        if done:
            break

all_rewards = np.array(all_rewards)
print(f"\nAggregated rewards from 11 environments:")
print(f"  Total steps: {len(all_rewards)}")
print(f"  Min: {all_rewards.min():.4f}")
print(f"  Max: {all_rewards.max():.4f}")
print(f"  Mean: {all_rewards.mean():.4f}")
print(f"  Std: {all_rewards.std():.4f}")

print(f"\nObservation stats across 11 environments:")
obs_mins = [s['min'] for s in all_obs_stats]
obs_maxs = [s['max'] for s in all_obs_stats]
obs_means = [s['mean'] for s in all_obs_stats]
print(f"  Min range: {np.min(obs_mins):.4f} to {np.max(obs_mins):.4f}")
print(f"  Max range: {np.min(obs_maxs):.4f} to {np.max(obs_maxs):.4f}")
print(f"  Mean range: {np.min(obs_means):.4f} to {np.max(obs_means):.4f}")
