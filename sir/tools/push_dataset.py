#!/usr/bin/env python3
"""
Push a local LeRobot dataset to HuggingFace Hub.

Usage:
    python -m sir.tools.push_dataset --repo-id <hub-name> --root <dataset-path> [options]

Examples:
    # Push local dataset to HuggingFace with a specific name
    python -m sir.tools.push_dataset --repo-id lift_minimal_state --root ./data/gripper_wait

    # Push as private dataset
    python -m sir.tools.push_dataset --repo-id my-private-dataset --root ./data/my_demos --private

Note: The --root should point to the full dataset directory (containing data/, meta/, videos/, etc.)
      The --repo-id is the name you want to use on HuggingFace Hub (can differ from local folder name)
"""

import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from huggingface_hub import HfApi


def main():
    parser = argparse.ArgumentParser(description="Push a local LeRobot dataset to HuggingFace Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository ID on HuggingFace Hub (e.g., 'username/dataset-name' or 'dataset-name')",
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Full path to the dataset directory (should contain data/, meta/, videos/, etc.)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on HuggingFace Hub",
    )
    parser.add_argument(
        "--branch",
        type=str,
        default=None,
        help="Git branch to push to (default: uses dataset version)",
    )

    args = parser.parse_args()

    # Get HuggingFace username and format repo_id
    print("Checking HuggingFace authentication...")
    hub_api = HfApi()
    user_info = hub_api.whoami()
    username = user_info["name"]

    # If repo_id doesn't contain a slash, prepend username
    repo_id = args.repo_id
    if "/" not in repo_id:
        repo_id = f"{username}/{repo_id}"
        print(f"Using full repo ID: {repo_id}")

    # Resolve paths
    dataset_path = Path(args.root)

    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    # Verify it's a valid LeRobot dataset
    required_dirs = ["data", "meta"]
    for dir_name in required_dirs:
        if not (dataset_path / dir_name).exists():
            raise FileNotFoundError(
                f"Invalid LeRobot dataset: missing '{dir_name}/' directory in {dataset_path}"
            )

    # Load the dataset
    dataset = LeRobotDataset(
        repo_id=repo_id,
        root=str(dataset_path),
    )

    print(f"\nDataset info:")
    print(f"  Episodes: {dataset.num_episodes}")
    print(f"  Frames: {dataset.num_frames}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Features: {list(dataset.features.keys())}")

    # Push to hub
    print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
    print(f"  Private: {args.private}")
    if args.branch:
        print(f"  Branch: {args.branch}")

    dataset.push_to_hub(
        branch=args.branch,
        private=args.private,
    )

    print(f"\n✓ Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
