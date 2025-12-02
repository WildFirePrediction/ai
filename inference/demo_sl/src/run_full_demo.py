"""
Full demo pipeline: Generate dummy fire -> Run inference -> Visualize
End-to-end test of the inference system
"""
import sys
from pathlib import Path
import subprocess
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))


def run_command(cmd, description):
    """Run a command and print output"""
    print(f"\n{'=' * 80}")
    print(f"{description}")
    print(f"{'=' * 80}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=str(project_root))

    if result.returncode != 0:
        print(f"\nError: Command failed with exit code {result.returncode}")
        return False

    return True


def main():
    """Run full demo pipeline"""
    parser = argparse.ArgumentParser(
        description='Full demo pipeline for wildfire inference'
    )
    parser.add_argument(
        '--num-fires',
        type=int,
        default=1,
        help='Number of fires to generate (default: 1)'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Use random locations instead of known fire zones'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device for inference (cuda or cpu)'
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("WILDFIRE INFERENCE - FULL DEMO PIPELINE")
    print("=" * 80)
    print(f"Number of fires: {args.num_fires}")
    print(f"Location mode: {'Random' if args.random else 'Known fire zones'}")
    print(f"Device: {args.device}")

    # Get Python interpreter
    python_exe = sys.executable

    # Step 1: Generate dummy fire
    cmd_generate = [
        python_exe,
        'inference/demo/src/generate_dummy_fire.py',
        '--num-fires', str(args.num_fires),
        '--output-dir', 'inference/demo/input'
    ]
    if args.random:
        cmd_generate.append('--random')

    if not run_command(cmd_generate, "STEP 1: Generating Dummy Fire Data"):
        return

    # Find the generated file (most recent)
    input_dir = project_root / 'inference/demo/input'
    input_files = sorted(input_dir.glob('dummy_fire_*.json'))
    if not input_files:
        print("Error: No dummy fire file generated")
        return

    input_file = input_files[-1]  # Most recent
    print(f"\nUsing input file: {input_file}")

    # Step 2: Run inference
    cmd_inference = [
        python_exe,
        'inference/demo/src/run_demo_inference.py',
        '--input', str(input_file),
        '--output-dir', 'inference/sl/outputs/production',
        '--device', args.device
    ]

    if not run_command(cmd_inference, "STEP 2: Running Inference Pipeline"):
        return

    # Find the generated prediction files
    output_dir = project_root / 'inference/sl/outputs/production'
    prediction_files = sorted(output_dir.glob('fire_*.json'))
    if not prediction_files:
        print("Error: No prediction files generated")
        return

    # Get the most recent prediction files (based on num_fires)
    recent_predictions = prediction_files[-args.num_fires:]
    print(f"\nFound {len(recent_predictions)} prediction file(s)")

    # Step 3: Visualize each prediction
    for i, pred_file in enumerate(recent_predictions):
        cmd_visualize = [
            python_exe,
            'inference/demo/src/visualize_prediction.py',
            '--input', str(pred_file),
            '--output', f'inference/demo/output/fire_{i+1}.html'
        ]

        if not run_command(cmd_visualize, f"STEP 3.{i+1}: Creating Visualization"):
            continue

    # Final summary
    print("\n" + "=" * 80)
    print("DEMO PIPELINE COMPLETE")
    print("=" * 80)

    print(f"\nGenerated files:")
    print(f"  Input: {input_file}")
    print(f"  Predictions:")
    for pred_file in recent_predictions:
        print(f"    {pred_file}")
    print(f"  Visualizations:")
    viz_dir = project_root / 'inference/demo/output'
    viz_files = sorted(viz_dir.glob('fire_*.html'))
    for viz_file in viz_files[-args.num_fires:]:
        print(f"    {viz_file}")

    print(f"\nOpen the HTML files in a browser to view the predictions!")


if __name__ == '__main__':
    main()
