"""
manifest_utils.py

Scans the outputs directory for each ticker/date and generates a reflexivity_manifest.json mapping:
{
  "SPY": {
    "2025-06-24": {
      "metrics": "outputs/SPY/2025-06-24/2025-06-24_metrics.json",
      "plot": "outputs/SPY/2025-06-24/gamma_bowl_SPY_2025-06-24.png",
      "strategy": "outputs/SPY/2025-06-24/strategy_signals_SPY_2025-06-24.md"
    },
    ...
  },
  ...
}

Usage:
    python manifest_utils.py --outputs_dir outputs --manifest_path reflexivity_manifest.json
"""
import os
import json
import argparse
from pathlib import Path

def generate_manifest(outputs_dir, manifest_path):
    manifest = {}
    outputs_dir = Path(outputs_dir)
    for ticker_dir in outputs_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        manifest[ticker] = {}
        for date_dir in ticker_dir.iterdir():
            if not date_dir.is_dir():
                continue
            date = date_dir.name
            metrics = None
            plot = None
            strategy = None
            for file in date_dir.iterdir():
                if file.name.endswith('_metrics.json'):
                    metrics = str(file)
                elif file.name.startswith('gamma_bowl') and file.suffix == '.png':
                    plot = str(file)
                elif file.name.startswith('strategy_signals') and file.suffix == '.md':
                    strategy = str(file)
            if metrics and plot and strategy:
                manifest[ticker][date] = {
                    "metrics": metrics,
                    "plot": plot,
                    "strategy": strategy
                }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] Manifest written to {manifest_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate reflexivity_manifest.json from outputs directory.")
    parser.add_argument('--outputs_dir', type=str, default='outputs', help='Outputs directory')
    parser.add_argument('--manifest_path', type=str, default='reflexivity_manifest.json', help='Manifest output path')
    args = parser.parse_args()
    generate_manifest(args.outputs_dir, args.manifest_path) 