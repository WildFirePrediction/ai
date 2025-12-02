"""
Full SL multi-point demo pipeline orchestrator
Generates multi-point dummy fire -> Runs SL inference -> Creates visualization
"""
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def run_full_sl_multi_demo(num_fires=1, device='cuda'):
    """
    Run complete SL multi-point demo pipeline

    Args:
        num_fires: Number of dummy fires to generate
        device: Device for inference

    Returns:
        list of (fire_id, html_path) tuples
    """
    demo_root = Path(__file__).parent.parent
    input_dir = demo_root / 'input'
    output_dir = demo_root / 'output'

    # Ensure directories exist
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("SL MULTI-POINT DEMO PIPELINE")
    print("="*70)
    print(f"Generating {num_fires} multi-point dummy fire(s) in Gyeongsangbuk-do...")
    print(f"Initial fire pattern: 1 center + 2 spread points (3 total cells)")
    print(f"Device: {device}")
    print("="*70)

    results = []

    for i in range(num_fires):
        print(f"\n{'='*70}")
        print(f"FIRE {i+1}/{num_fires}")
        print(f"{'='*70}")

        # Step 1: Generate multi-point dummy fire
        print("\nStep 1/3: Generating multi-point dummy fire...")
        dummy_fire_script = demo_root / 'src' / 'generate_dummy_fire.py'

        cmd = [
            sys.executable,
            str(dummy_fire_script),
            '--output-dir', str(input_dir),
            '--num-fires', '1'
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error generating dummy fire: {result.stderr}")
            continue

        # Find the most recently generated fire JSON
        fire_jsons = sorted(input_dir.glob('dummy_fire_multi_*.json'), key=lambda p: p.stat().st_mtime)
        if not fire_jsons:
            print("Error: No dummy fire JSON generated")
            continue

        dummy_fire_json = fire_jsons[-1]
        print(f"✓ Multi-point dummy fire generated: {dummy_fire_json}")

        # Step 2: Run SL inference
        print("\nStep 2/3: Running SL inference with multi-point initialization...")
        inference_script = demo_root / 'src' / 'run_demo_inference.py'

        cmd = [
            sys.executable,
            str(inference_script),
            '--input', str(dummy_fire_json),
            '--output-dir', str(output_dir),
            '--device', device
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running inference: {result.stderr}")
            continue

        # Find the prediction JSON (newest file matching pattern)
        prediction_jsons = sorted(output_dir.glob('fire_*.json'), key=lambda p: p.stat().st_mtime)
        if not prediction_jsons:
            print("Error: No prediction JSON found")
            continue

        prediction_json = prediction_jsons[-1]
        print(f"✓ SL inference complete: {prediction_json}")

        # Step 3: Visualize
        print("\nStep 3/3: Creating visualization...")
        visualize_script = demo_root / 'src' / 'visualize_prediction.py'
        html_output = output_dir / f'{prediction_json.stem}_map.html'

        cmd = [
            sys.executable,
            str(visualize_script),
            '--input', str(prediction_json),
            '--output', str(html_output)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error creating visualization: {result.stderr}")
            continue

        print(f"✓ Visualization created: {html_output}")

        # Extract fire_id from JSON name
        fire_id = prediction_json.stem.split('_')[1]
        results.append((fire_id, html_output))

        print(f"\n{'='*70}")
        print(f"FIRE {i+1} COMPLETE")
        print(f"  Prediction: {prediction_json}")
        print(f"  Map: {html_output}")
        print(f"{'='*70}")

    # Final summary
    print(f"\n{'='*70}")
    print("SL MULTI-POINT DEMO PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Processed {len(results)}/{num_fires} fires successfully")
    print(f"\nGenerated maps:")
    for fire_id, html_path in results:
        print(f"  Fire {fire_id}: {html_path}")
    print(f"{'='*70}")

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='SL Multi-Point Full Demo Pipeline')
    parser.add_argument('--num-fires', type=int, default=1,
                       help='Number of dummy fires to generate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device for inference (cuda/cpu)')

    args = parser.parse_args()

    results = run_full_sl_multi_demo(
        num_fires=args.num_fires,
        device=args.device
    )

    if results:
        print(f"\n✓ Successfully generated {len(results)} fire prediction map(s)")
        print("Open HTML files in browser to view results")
    else:
        print("\n✗ No successful predictions")
        sys.exit(1)
