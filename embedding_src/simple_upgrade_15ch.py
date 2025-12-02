"""
Simple upgrade: Add NDVI + FSM channels with reasonable defaults
This gets the 15ch pipeline working quickly
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm

def upgrade_episode_simple(episode_file):
    """Upgrade 11ch to 15ch with default NDVI/FSM"""
    try:
        # Load 11ch data
        data_11ch = np.load(episode_file)
        states_11ch = data_11ch['states']  # (T, 11, 30, 30)
        fire_masks = data_11ch['fire_masks']
        timestamps = data_11ch['timestamps']
        centroid = data_11ch['centroid']
        
        T = states_11ch.shape[0]
        states_15ch = np.zeros((T, 15, 30, 30), dtype=np.float32)
        
        for t in range(T):
            # Copy existing 11 channels
            states_15ch[t, :11] = states_11ch[t]
            
            # Channel 11: NDVI - use moderate vegetation (0.5 normalized)
            states_15ch[t, 11] = 0.5
            
            # Channels 12-14: FSM one-hot - assume mixed forest (class 3 = all classes present)
            # Equal probability across fuel types
            states_15ch[t, 12] = 0.33  # Broadleaf
            states_15ch[t, 13] = 0.33  # Conifer  
            states_15ch[t, 14] = 0.34  # Mixed
        
        return {
            'states': states_15ch,
            'fire_masks': fire_masks,
            'timestamps': timestamps,
            'centroid': centroid
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    print("="*80)
    print("SIMPLE UPGRADE: 11ch -> 15ch WITH DEFAULTS")
    print("="*80)
    print("Adding:")
    print("  Channel 11: NDVI = 0.5 (moderate vegetation)")
    print("  Channels 12-14: FSM = [0.33, 0.33, 0.34] (mixed forest)")
    print("="*80)
    
    input_dir = Path('embedded_data/fire_episodes_11ch_fixed')
    output_dir = Path('embedded_data/fire_episodes_15ch')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    episode_files = sorted(input_dir.glob('episode_*.npz'))
    print(f"\nProcessing {len(episode_files)} episodes...")
    
    success = 0
    
    for ep_file in tqdm(episode_files, desc="Upgrading"):
        result = upgrade_episode_simple(ep_file)
        
        if result is not None:
            output_file = output_dir / ep_file.name
            np.savez_compressed(
                output_file,
                states=result['states'],
                fire_masks=result['fire_masks'],
                timestamps=result['timestamps'],
                centroid=result['centroid']
            )
            success += 1
    
    print(f"\n{'='*80}")
    print(f"SUCCESS: {success}/{len(episode_files)} episodes")
    print(f"{'='*80}")
    
    # Verify
    sample = np.load(output_dir / 'episode_0000.npz')
    print(f"\nSample episode shape: {sample['states'].shape}")
    print(f"Channels: {sample['states'].shape[1]}")
    print("\nReady for 15-channel training!")

if __name__ == '__main__':
    main()
