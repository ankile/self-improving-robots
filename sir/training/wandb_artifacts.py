"""
W&B Artifact Management for LeRobot Policies

This module provides utilities for uploading and downloading model checkpoints
to/from Weights & Biases, ensuring full compatibility with LeRobot's
save_pretrained and from_pretrained mechanisms.

This is essential for:
1. Version control of trained models
2. Reproducibility across runs
3. Loading BC checkpoints for DAgger training
4. Easy model sharing between machines (local dev -> cluster)

The artifact includes all necessary files:
- model.safetensors (weights)
- config.json (policy configuration)
- policy_preprocessor.json + policy_preprocessor_step_*.safetensors (input normalization)
- policy_postprocessor.json + policy_postprocessor_step_*.safetensors (output denormalization)
"""

import json
import logging
import shutil
from pathlib import Path
from typing import Optional

import wandb

logger = logging.getLogger(__name__)


def get_checkpoint_files(checkpoint_path: Path) -> list[Path]:
    """
    Get all relevant checkpoint files that should be uploaded to W&B.

    These files form a complete LeRobot policy checkpoint that can be loaded
    with `ACTPolicy.from_pretrained(artifact_path)`.

    Args:
        checkpoint_path: Path to the checkpoint directory

    Returns:
        List of Path objects for all checkpoint files that exist
    """
    # Core policy files (always present)
    core_files = [
        "config.json",  # Policy configuration
        "model.safetensors",  # Model weights in SafeTensors format
    ]

    # Processor files (pipeline configs and statistics)
    # These are created by preprocessor.save_pretrained() and postprocessor.save_pretrained()
    processor_files = [
        "policy_preprocessor.json",  # Input normalization pipeline config
        "policy_postprocessor.json",  # Output denormalization pipeline config
    ]

    # Dynamic processor files (named with step index)
    # These contain the actual normalization statistics (mean/std)
    dynamic_files = []
    for file_path in checkpoint_path.glob("policy_*_step_*.safetensors"):
        dynamic_files.append(file_path.name)

    # Collect all files that exist
    all_files = core_files + processor_files + dynamic_files
    existing_files = []

    for filename in all_files:
        file_path = checkpoint_path / filename
        if file_path.exists():
            existing_files.append(file_path)

    if not existing_files:
        logger.warning(f"No checkpoint files found in {checkpoint_path}")

    return existing_files


def upload_checkpoint_to_wandb(
    checkpoint_path: Path,
    artifact_name: str,
    artifact_type: str = "policy",
    description: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> Optional[wandb.Artifact]:
    """
    Upload a complete LeRobot policy checkpoint to W&B as an artifact.

    This uploads all necessary files for loading the policy later with:
    `policy, preprocessor, postprocessor = load_checkpoint_from_wandb(...)`

    Args:
        checkpoint_path: Path to the checkpoint directory (containing config.json, model.safetensors, etc.)
        artifact_name: Name for the W&B artifact (will be sanitized)
        artifact_type: Type of artifact (default: "policy")
        description: Optional description for the artifact
        metadata: Optional metadata dict to attach to the artifact

    Returns:
        The uploaded wandb.Artifact object, or None if not logged in to W&B

    Example:
        >>> checkpoint_path = Path("checkpoints/best_model")
        >>> artifact = upload_checkpoint_to_wandb(
        ...     checkpoint_path,
        ...     artifact_name=f"act-lift-best",
        ...     description=f"Best ACT policy for Lift task (success: 95%)",
        ...     metadata={"task": "Lift", "success_rate": 0.95}
        ... )
    """
    if not wandb.run:
        logger.warning("W&B run not active. Skipping artifact upload.")
        return None

    # Sanitize artifact name for W&B compatibility
    artifact_name = _sanitize_artifact_name(artifact_name)

    logger.info(f"Creating W&B artifact: {artifact_name}")
    artifact = wandb.Artifact(
        name=artifact_name,
        type=artifact_type,
        description=description,
        metadata=metadata,
    )

    # Get all checkpoint files
    checkpoint_files = get_checkpoint_files(checkpoint_path)

    if not checkpoint_files:
        logger.error(f"No checkpoint files found in {checkpoint_path}")
        return None

    # Add each file to the artifact
    logger.info(f"Adding {len(checkpoint_files)} files to artifact")
    for file_path in checkpoint_files:
        # Use just the filename as the artifact path (preserves directory structure)
        artifact.add_file(str(file_path), name=file_path.name)
        logger.debug(f"  Added: {file_path.name}")

    # Log the artifact
    logger.info(f"Uploading artifact to W&B...")
    wandb.log_artifact(artifact)
    logger.info(f"✓ Artifact '{artifact_name}' uploaded successfully")

    return artifact


def download_checkpoint_from_wandb(
    artifact_identifier: str,
    download_dir: Optional[Path] = None,
) -> Path:
    """
    Download a policy checkpoint from W&B artifact.

    The checkpoint includes all files needed to reconstruct the policy:
    - model.safetensors (weights)
    - config.json (architecture)
    - policy_preprocessor.json + .safetensors (input normalization)
    - policy_postprocessor.json + .safetensors (output normalization)

    Args:
        artifact_identifier: Artifact identifier in format "entity/project/name:version" or "name:version"
        download_dir: Directory to download to (default: uses cache)

    Returns:
        Path to the downloaded checkpoint directory

    Example:
        >>> # Download best checkpoint for DAgger
        >>> checkpoint_path = download_checkpoint_from_wandb(
        ...     artifact_identifier="self-improving/act-training/act-lift-best:latest"
        ... )
        >>> # Load for DAgger training
        >>> from lerobot.policies.act.modeling_act import ACTPolicy
        >>> policy = ACTPolicy.from_pretrained(str(checkpoint_path))
    """
    logger.info(f"Downloading artifact from W&B: {artifact_identifier}")

    # Use run.use_artifact() if we have an active run (proper W&B lineage tracking)
    # Otherwise fall back to Api().artifact() (only requires being logged in)
    if wandb.run:
        logger.info("Using run.use_artifact() to track artifact usage in this run")
        artifact = wandb.run.use_artifact(artifact_identifier)
    else:
        logger.info("No active run, using Api().artifact()")
        api = wandb.Api()
        try:
            artifact = api.artifact(artifact_identifier)
        except Exception as e:
            logger.error(f"Failed to find artifact '{artifact_identifier}': {e}")
            raise

    # Create download directory
    if download_dir is None:
        download_dir = Path.home() / ".cache" / "sir_wandb_artifacts" / artifact.name
    else:
        download_dir = Path(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading to: {download_dir}")
    artifact_dir = artifact.download(root=str(download_dir))

    return Path(artifact_dir)


def list_checkpoints_from_wandb(
    artifact_type: str = "policy",
    project: Optional[str] = None,
) -> list[dict]:
    """
    List available policy checkpoints in W&B.

    Args:
        artifact_type: Type of artifact to list (default: "policy")
        project: Project name (uses current run's project if not specified)

    Returns:
        List of dicts with artifact metadata

    Example:
        >>> checkpoints = list_checkpoints_from_wandb()
        >>> for ckpt in checkpoints:
        ...     print(f"{ckpt['name']}: {ckpt['description']}")
    """
    if not wandb.run:
        logger.error("W&B run not active. Cannot list artifacts.")
        return []

    if project is None:
        project = wandb.run.project

    logger.info(f"Listing artifacts from project '{project}'")

    api = wandb.Api()
    try:
        artifacts = api.artifacts(
            type_name=artifact_type,
            project=project,
        )
        return [
            {
                "name": a.name,
                "version": a.version,
                "description": a.description,
                "metadata": a.metadata,
                "created_at": a.created_at,
            }
            for a in artifacts
        ]
    except Exception as e:
        logger.error(f"Failed to list artifacts: {e}")
        return []


def log_checkpoint_metrics(
    checkpoint_path: Path,
    metrics: dict,
    step: Optional[int] = None,
    prefix: str = "checkpoint",
) -> None:
    """
    Log metrics associated with a checkpoint to W&B.

    Args:
        checkpoint_path: Path to checkpoint for reference
        metrics: Dictionary of metrics (e.g., success_rate, avg_reward)
        step: Training step number
        prefix: Prefix for metric names

    Example:
        >>> metrics = {"success_rate": 0.95, "avg_reward": 0.87}
        >>> log_checkpoint_metrics(
        ...     Path("checkpoints/best_model"),
        ...     metrics,
        ...     step=5000,
        ...     prefix="eval"
        ... )
    """
    if not wandb.run:
        return

    log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}

    if step is not None:
        wandb.log(log_dict, step=step)
    else:
        wandb.log(log_dict)

    logger.debug(f"Logged metrics: {log_dict}")


def _sanitize_artifact_name(name: str) -> str:
    """
    Sanitize artifact name for W&B compatibility.

    W&B has restrictions on artifact names:
    - Must contain only alphanumeric characters, dashes, underscores, and dots
    - No spaces or special characters
    """
    import re

    # Replace invalid characters with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Remove leading/trailing underscores
    sanitized = sanitized.strip("_")

    return sanitized


def create_artifact_metadata(
    repo_id: str,
    success_rate: float,
    avg_reward: float,
    step: int,
    is_best: bool = False,
    dataset_size: int = 0,
    checkpoint_index: Optional[int] = None,
) -> dict:
    """
    Create standardized metadata dict for artifact logging.

    Args:
        repo_id: Dataset repository ID
        success_rate: Evaluation success rate (0-1)
        avg_reward: Average episode reward
        step: Training step number
        is_best: Whether this is the best checkpoint
        dataset_size: Number of training frames
        checkpoint_index: Index if periodic checkpoint (e.g., 1st, 2nd periodic)

    Returns:
        Metadata dictionary suitable for W&B artifact
    """
    metadata = {
        "repo_id": repo_id,
        "success_rate": f"{success_rate:.3f}",
        "avg_reward": f"{avg_reward:.3f}",
        "step": step,
        "is_best": is_best,
        "dataset_size": dataset_size,
    }

    if checkpoint_index is not None:
        metadata["checkpoint_index"] = checkpoint_index

    return metadata
