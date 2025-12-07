"""
Batch visualizer for RL inference outputs
Converts JSON prediction files to HTML maps

Scans inference/rl/outputs/api/*.json and creates HTML maps in inference/rl/outputs/map/

USAGE:
    # Visualize all JSON files in api directory
    python inference/rl/visualizer.py

    # Visualize specific JSON file
    python inference/rl/visualizer.py --input inference/rl/outputs/api/fire_123.json

    # Custom output directory
    python inference/rl/visualizer.py --output-dir custom_maps/

    # Watch mode: continuously process new files
    python inference/rl/visualizer.py --watch
"""
import sys
from pathlib import Path
import json
import argparse
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Import visualization function from demo_rl
sys.path.insert(0, str(project_root / 'inference' / 'demo_rl' / 'src'))
from visualize_prediction import create_prediction_map


def visualize_json_file(json_path, output_dir):
    """
    Create HTML visualization for a single JSON file

    Args:
        json_path: Path to prediction JSON file
        output_dir: Directory to save HTML output

    Returns:
        output_path: Path to saved HTML file or None if failed
    """
    try:
        # Load JSON
        with open(json_path, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)

        # Get fire ID for filename
        fire_id = prediction_data.get('fire_id', 'unknown')
        fire_timestamp = prediction_data.get('fire_timestamp', '')

        # Create output filename
        json_filename = json_path.stem  # e.g., "fire_123_20251207_123456"
        output_filename = f"{json_filename}_map.html"
        output_path = output_dir / output_filename

        # Create visualization
        create_prediction_map(prediction_data, output_path)

        return output_path

    except Exception as e:
        print(f"  [ERROR] Failed to visualize {json_path.name}: {e}")
        return None


def scan_and_visualize(api_dir, output_dir, processed_files=None):
    """
    Scan API directory for JSON files and create visualizations

    Args:
        api_dir: Directory containing JSON files
        output_dir: Directory to save HTML outputs
        processed_files: Set of already processed files (for watch mode)

    Returns:
        newly_processed: Set of newly processed file paths
    """
    api_dir = Path(api_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if processed_files is None:
        processed_files = set()

    # Find all JSON files
    json_files = sorted(api_dir.glob('*.json'))

    if not json_files:
        print(f"No JSON files found in {api_dir}")
        return set()

    # Filter to only new files
    new_files = [f for f in json_files if f not in processed_files]

    if not new_files:
        return set()

    print(f"\n{'='*70}")
    print(f"Found {len(new_files)} new JSON file(s) to visualize")
    print(f"  Input:  {api_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}\n")

    newly_processed = set()
    success_count = 0

    for i, json_path in enumerate(new_files, 1):
        print(f"[{i}/{len(new_files)}] Processing {json_path.name}...")

        output_path = visualize_json_file(json_path, output_dir)

        if output_path:
            success_count += 1
            newly_processed.add(json_path)
            print(f"  [SUCCESS] Saved to {output_path.name}\n")
        else:
            print(f"  [FAILED] Could not process file\n")

    print(f"{'='*70}")
    print(f"Visualization complete: {success_count}/{len(new_files)} succeeded")
    print(f"{'='*70}\n")

    return newly_processed


def watch_mode(api_dir, output_dir, interval=10):
    """
    Watch mode: continuously monitor for new JSON files

    Args:
        api_dir: Directory to watch for JSON files
        output_dir: Directory to save HTML outputs
        interval: Check interval in seconds
    """
    print("="*70)
    print("WATCH MODE - Continuously monitoring for new JSON files")
    print("="*70)
    print(f"Input directory:  {api_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Check interval:   {interval}s")
    print(f"\nPress Ctrl+C to stop\n")
    print("="*70)

    processed_files = set()
    iteration = 0

    try:
        while True:
            iteration += 1
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{timestamp}] Check #{iteration}")

            newly_processed = scan_and_visualize(api_dir, output_dir, processed_files)
            processed_files.update(newly_processed)

            if not newly_processed:
                print(f"  No new files. Sleeping for {interval}s...")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n[STOPPING] Watch mode interrupted by user")
        print(f"Total files processed: {len(processed_files)}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Batch visualizer for RL inference outputs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all JSON files
  python inference/rl/visualizer.py

  # Visualize specific file
  python inference/rl/visualizer.py --input inference/rl/outputs/api/fire_123.json

  # Watch mode (continuous monitoring)
  python inference/rl/visualizer.py --watch

  # Custom directories
  python inference/rl/visualizer.py --api-dir custom/api --output-dir custom/maps
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        help='Visualize single JSON file (overrides --api-dir)'
    )
    parser.add_argument(
        '--api-dir',
        type=str,
        default='inference/rl/outputs/api',
        help='Directory containing JSON files (default: inference/rl/outputs/api)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='inference/rl/outputs/map',
        help='Directory to save HTML maps (default: inference/rl/outputs/map)'
    )
    parser.add_argument(
        '--watch',
        action='store_true',
        help='Watch mode: continuously monitor for new files'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=10,
        help='Watch mode check interval in seconds (default: 10)'
    )

    args = parser.parse_args()

    # Single file mode
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"[ERROR] Input file not found: {input_path}")
            return 1

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Visualizing single file: {input_path}")
        output_path = visualize_json_file(input_path, output_dir)

        if output_path:
            print(f"\n[SUCCESS] Visualization saved to {output_path}")
            return 0
        else:
            print(f"\n[FAILED] Could not visualize file")
            return 1

    # Watch mode
    elif args.watch:
        api_dir = Path(args.api_dir)
        output_dir = Path(args.output_dir)

        if not api_dir.exists():
            print(f"[ERROR] API directory not found: {api_dir}")
            print(f"Creating directory: {api_dir}")
            api_dir.mkdir(parents=True, exist_ok=True)

        watch_mode(api_dir, output_dir, args.interval)
        return 0

    # Batch mode (default)
    else:
        api_dir = Path(args.api_dir)
        output_dir = Path(args.output_dir)

        if not api_dir.exists():
            print(f"[ERROR] API directory not found: {api_dir}")
            return 1

        scan_and_visualize(api_dir, output_dir)
        return 0


if __name__ == '__main__':
    sys.exit(main())
